"""Run Claude Code as a parallel deciphering reader and merge its votes.

This uses the installed `claude` CLI in non-interactive print mode. It reads
the existing plate images from predictions/decipher/{seg_id}/plates, writes a
manual Claude file, recomputes consensus, and mirrors result.json to the web app.

Examples:
    python scripts/claude_code_decipher.py --only 20231221180251
    python scripts/claude_code_decipher.py --only 20231221180251 --parallel 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decipher.alignment import align_result_characters, load_detection_lines  # noqa: E402
from decipher.consensus import build_consensus  # noqa: E402
from decipher.pipeline import _segment_summary  # noqa: E402

PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"
MODEL_SLUG = "manual/claude-opus-4.7"
MODEL_NAME = "Claude Code"


def _claude_exe() -> str:
    exe = shutil.which("claude.cmd") or shutil.which("claude") or shutil.which("claude.ps1")
    if not exe:
        raise FileNotFoundError("Could not find Claude Code CLI (`claude`) on PATH")
    return exe


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Claude did not return a JSON object: {text[:200]}")
    return json.loads(text[start:end + 1])


def _normalise_reading(parsed: Dict[str, Any]) -> Dict[str, Any]:
    chars = parsed.get("characters") or []
    out_chars = []
    for i, c in enumerate(chars):
        if not isinstance(c, dict):
            continue
        ch = str(c.get("char") or "").strip()
        if not ch:
            continue
        try:
            x_norm = float(c.get("x_norm", (i + 0.5) / max(1, len(chars))))
        except (TypeError, ValueError):
            x_norm = (i + 0.5) / max(1, len(chars))
        try:
            confidence = float(c.get("confidence", 0.45))
        except (TypeError, ValueError):
            confidence = 0.45
        out_chars.append({
            "char": ch[0],
            "x_norm": max(0.0, min(1.0, x_norm)),
            "confidence": max(0.0, min(1.0, confidence)),
            "alternatives": c.get("alternatives") if isinstance(c.get("alternatives"), list) else [],
        })

    line_text = str(parsed.get("line_text") or "").strip()
    if not line_text and out_chars:
        line_text = "".join(c["char"] for c in out_chars)

    return {
        "line_text": line_text,
        "characters": out_chars,
        "translation_en": str(parsed.get("translation_en") or "[uncertain]").strip() or "[uncertain]",
        "probable_summary": str(parsed.get("probable_summary") or "[uncertain]").strip() or "[uncertain]",
        "notes": str(parsed.get("notes") or "").strip(),
        "overall_confidence": str(parsed.get("overall_confidence") or "low").strip().lower(),
    }


def _prompt(seg_id: str, strip_id: int, plate_path: Path) -> str:
    rel = plate_path.relative_to(ROOT)
    return f"""
You are reading a Herculaneum papyrus ink-probability plate.

Inspect this local image path carefully:
{rel.as_posix()}

Return ONLY valid JSON with this exact shape:
{{
  "line_text": "Greek majuscule/lunate-sigma transcription, use . for gaps",
  "characters": [
    {{"char": "Α", "x_norm": 0.123, "confidence": 0.55, "alternatives": ["Λ", "Δ"]}}
  ],
  "translation_en": "[uncertain] or a short translation",
  "probable_summary": "short statement of what the line probably says",
  "notes": "visual caveats; mention if the strip is mostly noise",
  "overall_confidence": "low|medium|high"
}}

Rules:
- Do not invent a fluent reading when the image is fragmentary.
- Prefer ancient Greek majuscule forms. Use Ϲ for lunate sigma if visible.
- x_norm must be 0..1 from left to right across the image.
- Segment {seg_id}, strip {strip_id}.
""".strip()


async def _run_claude_for_plate(seg_id: str, strip_id: int, plate_path: Path, model: str, budget: float) -> Dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        _claude_exe(), "-p",
        "--model", model,
        "--max-budget-usd", f"{budget:.2f}",
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(ROOT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ROOT),
    )
    stdout, stderr = await proc.communicate(_prompt(seg_id, strip_id, plate_path).encode("utf-8"))
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        out = stdout.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or out or f"claude exited {proc.returncode}")
    return _normalise_reading(_extract_json(stdout.decode("utf-8", errors="replace")))


async def _gather_readings(seg_id: str, plate_paths: Iterable[Path], model: str, parallel: int, budget: float) -> Dict[str, Dict[str, Any]]:
    sem = asyncio.Semaphore(parallel)
    out: Dict[str, Dict[str, Any]] = {}

    async def one(path: Path) -> None:
        strip_id = int(path.stem.rsplit("_", 1)[-1])
        async with sem:
            print(f"[claude] {seg_id} strip {strip_id}: reading {path.name}")
            out[str(strip_id)] = await _run_claude_for_plate(seg_id, strip_id, path, model, budget)

    await asyncio.gather(*(one(p) for p in plate_paths))
    return out


def _merge(seg_id: str, readings: Dict[str, Dict[str, Any]]) -> None:
    seg_dir = PRED_ROOT / seg_id
    result_path = seg_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing {result_path}; run scripts/decipher_all_segments.py first")
    data = json.loads(result_path.read_text(encoding="utf-8"))

    manual_dir = seg_dir / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    (manual_dir / "claude-opus-4.7.json").write_text(
        json.dumps({"model": MODEL_SLUG, "strips": readings}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for strip in data.get("strips", []):
        sid = str(strip.get("strip_id"))
        parsed = readings.get(sid)
        if not parsed:
            continue
        strip.setdefault("per_model", {})[MODEL_SLUG] = {
            "tier": 0,
            "display_name": MODEL_NAME,
            "raw": json.dumps(parsed, ensure_ascii=False),
            "parsed": parsed,
            "error": None,
            "finish_reason": "claude_code",
            "usage": None,
        }
        per_model_parsed = {
            slug: entry.get("parsed")
            for slug, entry in strip.get("per_model", {}).items()
            if isinstance(entry, dict)
        }
        strip["consensus"] = build_consensus(per_model_parsed).to_dict()

    manual = list(data.get("models_used_manual", []))
    if MODEL_SLUG not in manual:
        manual.append(MODEL_SLUG)
    data["models_used_manual"] = manual
    summary = _segment_summary(data.get("strips", []))
    data["segment_summary"] = summary
    data["segment_text"] = summary["text"]
    data["segment_meaning"] = summary["probable_summary"]
    data["segment_translation_en"] = summary["english_translation"]
    data["probable_scroll_summary"] = summary["probable_scroll_summary"]

    data = align_result_characters(
        data,
        strips_dir=seg_dir / "strips",
        detection_lines=load_detection_lines(seg_dir / "result.detection.json"),
    )
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    web_dir = WEB_ROOT / seg_id
    web_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, web_dir / "result.json")
    if (seg_dir / "manual").exists():
        web_manual = web_dir / "manual"
        if web_manual.exists():
            shutil.rmtree(web_manual)
        shutil.copytree(seg_dir / "manual", web_manual)
    print(f"[merge] wrote {result_path} and mirrored to {web_dir}")


async def main_async(args: argparse.Namespace) -> None:
    seg_ids = [args.only] if args.only else [p.name for p in sorted(PRED_ROOT.iterdir()) if (p / "result.json").exists()]
    for seg_id in seg_ids:
        plates_dir = PRED_ROOT / seg_id / "plates"
        plate_paths = sorted(plates_dir.glob("plate_*.png"))
        if args.max_strips:
            plate_paths = plate_paths[:args.max_strips]
        if not plate_paths:
            print(f"[skip] {seg_id}: no plates in {plates_dir}")
            continue
        try:
            readings = await _gather_readings(seg_id, plate_paths, args.model, args.parallel, args.budget_per_strip)
        except Exception as e:
            print(f"[error] {seg_id}: Claude Code run failed: {e}")
            continue
        _merge(seg_id, readings)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Run one segment id. Defaults to every prediction result.")
    p.add_argument("--model", default="sonnet", help="Claude Code model alias/name, e.g. sonnet or opus.")
    p.add_argument("--parallel", type=int, default=2, help="Concurrent Claude strip readings.")
    p.add_argument("--budget-per-strip", type=float, default=0.25)
    p.add_argument("--max-strips", type=int, default=0, help="Debug limit.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
