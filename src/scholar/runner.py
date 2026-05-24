"""Orchestrator for the Scholar stage.

Reads `predictions/decipher/{seg_id}/result.json`, runs the strip and
segment agents, and writes:

  - back into result.json under top-level `"scholar"` (additive)
  - a sibling `scholar.json` for clean diffing
  - mirrors both to `web/public/assets/decipher/{seg_id}/`
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# Ensure stdout can render Greek on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> str:
    """Strip markdown code fences and surrounding prose; return raw JSON substring."""
    if not text:
        return ""
    s = _FENCE_RE.sub("", text).strip()
    # If there is still prose, slice from first { to matching last }.
    if not s.startswith("{"):
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            s = s[i:j + 1]
    return s


def _parse_pydantic(raw: str, schema: Type[T]) -> T:
    js = _extract_json(raw)
    try:
        return schema.model_validate_json(js)
    except ValidationError:
        # Last-resort: parse as Python dict (handles single quotes etc.)
        return schema.model_validate(json.loads(js))

from .agent import (
    SegmentScholar,
    StripScholar,
    build_segment_agent,
    build_segment_input,
    build_strip_agent,
    build_strip_input,
)


ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"
AGENT_DECODING_ROOT = ROOT / "agent_decoding"


async def _run_strip(agent, strip: Dict[str, Any], seg_dir: Path) -> StripScholar:
    from agents import Runner

    strip_id = int(strip["strip_id"])
    image_path = seg_dir / strip.get("image_path", f"strips/strip_{strip_id:02d}.png")
    image_bytes = image_path.read_bytes()

    input_msgs = build_strip_input(
        strip_id=strip_id,
        image_bytes=image_bytes,
        consensus=strip.get("consensus") or {},
        template_hints=strip.get("template_hints") or [],
        per_model=strip.get("per_model") or {},
    )
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            result = await Runner.run(agent, input=input_msgs)
            raw = result.final_output if isinstance(result.final_output, str) else str(result.final_output or "")
            if not raw or not raw.strip():
                raise ValueError("empty model output")
            out = _parse_pydantic(raw, StripScholar)
            out.strip_id = strip_id
            return out
        except Exception as e:
            last_err = e
            if attempt == 0:
                await asyncio.sleep(1.5)
                continue
            raise
    raise last_err  # type: ignore[misc]


async def run_scholar_for_segment(
    seg_id: str,
    *,
    predictions_dir: Optional[Path] = None,
    mirror_to_web: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    pred_dir = Path(predictions_dir) if predictions_dir else PRED_ROOT
    seg_dir = pred_dir / seg_id
    result_path = seg_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} not found. Run scripts/decipher_all_segments.py first."
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    strips = result.get("strips") or []
    if not strips:
        raise ValueError(f"No strips in {result_path}")

    print(f"[scholar] {seg_id}: {len(strips)} strips")
    strip_agent = build_strip_agent()
    segment_agent = build_segment_agent()

    strip_scholars: List[StripScholar] = []
    for strip in strips:
        sid = strip.get("strip_id")
        print(f"  -> strip {sid} reading via strip-agent ...")
        try:
            s = await _run_strip(strip_agent, strip, seg_dir)
            strip_scholars.append(s)
            print(f"     letters={len(s.letters)}  paraphrase='{s.paraphrase_en[:60]}...'")
        except Exception as e:
            print(f"     FAILED: {e!r}")
            strip_scholars.append(StripScholar(
                strip_id=int(sid or 0),
                letters=[],
                word_divisions=[],
                recognized_words=[],
                paraphrase_en="",
                caveats=f"agent error: {e!r}",
            ))

    print(f"[scholar] {seg_id}: running segment synthesis ...")
    from agents import Runner
    seg_input = build_segment_input(strip_scholars)
    try:
        seg_result = await Runner.run(segment_agent, input=seg_input)
        raw = seg_result.final_output if isinstance(seg_result.final_output, str) else str(seg_result.final_output or "")
        segment_scholar = _parse_pydantic(raw, SegmentScholar)
    except Exception as e:
        print(f"  segment-agent FAILED: {e!r}")
        segment_scholar = SegmentScholar(
            probable_genre="unknown",
            lexical_signals=[],
            historical_context=f"Segment synthesis failed: {e!r}",
            candidate_authors=[],
            overall_paraphrase="",
            confidence_band="speculative",
        )

    scholar_block = {
        "strip_model": strip_agent.model.model if hasattr(strip_agent.model, "model") else "",
        "segment_model": segment_agent.model.model if hasattr(segment_agent.model, "model") else "",
        "segment": segment_scholar.model_dump(),
        "strips": [s.model_dump() for s in strip_scholars],
    }

    if dry_run:
        print(json.dumps(scholar_block, ensure_ascii=False, indent=2))
        return scholar_block

    # Persist
    result["scholar"] = scholar_block
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (seg_dir / "scholar.json").write_text(
        json.dumps(scholar_block, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Per-segment archive under agent_decoding/{seg_id}.json
    AGENT_DECODING_ROOT.mkdir(parents=True, exist_ok=True)
    (AGENT_DECODING_ROOT / f"{seg_id}.json").write_text(
        json.dumps(scholar_block, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[scholar] wrote {result_path} (with .scholar), scholar.json, agent_decoding/{seg_id}.json")

    if mirror_to_web:
        web_dir = WEB_ROOT / seg_id
        web_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, web_dir / "result.json")
        shutil.copy2(seg_dir / "scholar.json", web_dir / "scholar.json")
        print(f"[scholar] mirrored to {web_dir}")

    return scholar_block


def run_scholar_for_segment_sync(seg_id: str, **kw) -> Dict[str, Any]:
    return asyncio.run(run_scholar_for_segment(seg_id, **kw))
