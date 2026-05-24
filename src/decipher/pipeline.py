"""End-to-end deciphering pipeline for one segment.

  ink_labels_enhanced.tif
        │
        ▼  strip_extractor.extract_strips
  list[Strip] (with PNG bytes)
        │
        ▼  for each strip × each open-source model:
        │     openrouter_client.chat_with_image
        │  merge manual readings from {seg}/manual/*.json
        ▼
  consensus.build_consensus  → one ConsensusResult per strip
        │
        ▼
  result.json + strips/strip_NN.png
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alignment import align_result_characters, load_detection_lines
from .consensus import ConsensusResult, build_consensus
from .model_registry import MANUAL_MODELS, OPEN_SOURCE_MODELS, ModelSpec
from .openrouter_client import CompletionResult, OpenRouterClient
from .prompt_builder import SYSTEM_PROMPT, StripPromptContext, build_user_prompt
from .reading_plate import build_reading_plate
from .strip_extractor import Strip, extract_strips
from .template_hints import build_prompt_hints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment_summary(strips_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts: List[str] = []
    translations: List[str] = []
    summaries: List[str] = []
    n_chars = 0
    n_high = 0
    n_med = 0

    for strip in strips_out:
        consensus = strip.get("consensus", {})
        text = str(consensus.get("text") or "").strip()
        if text:
            texts.append(text)
        tr = str(consensus.get("translation_en") or "").strip()
        if tr and tr != "[uncertain]":
            translations.append(tr)
        sm = str(consensus.get("probable_summary") or "").strip()
        if sm and sm != "[uncertain]":
            summaries.append(sm)
        chars = consensus.get("characters") or []
        n_chars += len(chars)
        for c in chars:
            tier = c.get("tier") if isinstance(c, dict) else None
            if tier == "HIGH":
                n_high += 1
            elif tier == "MED":
                n_med += 1

    joined = "\n".join(texts)
    if summaries:
        probable = " / ".join(dict.fromkeys(summaries[:4]))
    elif translations:
        probable = " / ".join(dict.fromkeys(translations[:4]))
    elif texts:
        probable = (
            "Uncertain Greek majuscule fragments. The current ink map supports "
            "isolated letter hypotheses, but not a stable continuous reading."
        )
    else:
        probable = (
            "No reliable transcription yet. The visible ink labels are still too "
            "fragmentary for the current model ensemble."
        )

    if n_chars == 0:
        confidence = "none"
    else:
        anchor_frac = (n_high + 0.5 * n_med) / n_chars
        confidence = "medium" if anchor_frac >= 0.45 else "low"

    return {
        "text": joined,
        "probable_summary": probable,
        "english_translation": " / ".join(dict.fromkeys(translations[:6])) if translations else "[uncertain]",
        "probable_scroll_summary": probable,
        "confidence": confidence,
        "n_consensus_chars": n_chars,
        "n_high_confidence_chars": n_high,
        "n_medium_confidence_chars": n_med,
    }


def _consensus_from_template_hints(records: List[dict]) -> Dict[str, Any]:
    chars: List[Dict[str, Any]] = []
    for rec in sorted(records, key=lambda r: r.get("x_norm", 0.0)):
        candidates = rec.get("candidates") or []
        if not candidates:
            continue
        ch, prob = candidates[0]
        try:
            conf = float(prob)
        except (TypeError, ValueError):
            conf = 0.25
        chars.append({
            "char": ch,
            "x_norm": float(rec.get("x_norm", 0.0)),
            "confidence": round(min(conf, 0.45), 3),
            "tier": "LOW",
            "alternatives": [alt for alt, _ in candidates[1:3]],
            "votes": [{
                "model": "local/template_hints",
                "char": ch,
                "confidence": round(min(conf, 0.45), 3),
                "x_norm": float(rec.get("x_norm", 0.0)),
            }],
        })
    text = "".join(c["char"] for c in chars)
    return {
        "text": text,
        "translation_en": "[uncertain]",
        "probable_summary": "[uncertain]",
        "translation_votes": [],
        "summary_votes": [],
        "n_models_used": 0,
        "notes": (
            "Template-only fallback: no vision model reading was available; "
            "letters are low-confidence shape matches from visible ink blobs."
        ),
        "characters": chars,
    }

def _load_manual_readings(manual_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load any manually-pasted readings.

    File layout per segment:
        manual/<model_safe_id>.json
        {
          "model": "manual/claude-opus-4.7",
          "strips": {"0": { ...same schema as OpenRouter response... }, "1": {...}}
        }
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if not manual_dir.is_dir():
        return out
    for f in manual_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = data.get("model") or f"manual/{f.stem}"
        strips_obj = data.get("strips", {})
        by_idx: Dict[int, Dict[str, Any]] = {}
        for k, v in strips_obj.items():
            try:
                by_idx[int(k)] = v
            except (TypeError, ValueError):
                continue
        if by_idx:
            out[slug] = by_idx
    return out


async def _query_model_for_strip(
    client: OpenRouterClient,
    model: ModelSpec,
    strip: Strip,
    ctx: StripPromptContext,
    image_bytes: Optional[bytes] = None,
) -> CompletionResult:
    return await client.chat_with_image(
        model=model.slug,
        system_prompt=SYSTEM_PROMPT,
        user_text=build_user_prompt(ctx),
        image_bytes=image_bytes or strip.png_bytes,
        image_mime="image/png",
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def decipher_segment_async(
    *,
    seg_id: str,
    label_path: Path,
    out_dir: Path,
    n_strips: int = 8,
    strip_width: int = 3600,
    strip_height: int = 384,
    models: Optional[List[ModelSpec]] = None,
    skip_api: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the pipeline for one segment. Returns the full result dict."""
    label_path = Path(label_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strips_dir = out_dir / "strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    print(f"[decipher] {seg_id}: extracting strips from {label_path.name}")
    strips = extract_strips(
        label_path,
        strip_width=strip_width,
        strip_height=strip_height,
        n_strips=n_strips,
    )
    print(f"  -> {len(strips)} strips")

    for s in strips:
        (strips_dir / f"strip_{s.index:02d}.png").write_bytes(s.png_bytes)
    plates_dir = out_dir / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)
    plate_bytes_by_strip: Dict[int, bytes] = {}
    for s in strips:
        _, plate_png = build_reading_plate(s.image_u8)
        plate_bytes_by_strip[s.index] = plate_png
        (plates_dir / f"plate_{s.index:02d}.png").write_bytes(plate_png)

    # Pre-compute per-strip template-matching hints (probable-letter dictionary
    # + historical Greek/Epicurean context). Same hints are sent to every model.
    print(f"[decipher] building template hints for {len(strips)} strip(s) ...")
    hints_by_strip: Dict[int, tuple] = {}
    for s in strips:
        try:
            records, block = build_prompt_hints(s.png_bytes)
            hints_by_strip[s.index] = (records, block)
            print(f"  strip {s.index}: {len(records)} candidates above floor")
        except Exception as e:
            print(f"  strip {s.index}: hint build failed ({e!r}); continuing without hints")
            hints_by_strip[s.index] = ([], "")

    chosen_models = models if models is not None else OPEN_SOURCE_MODELS

    # ---- per-strip × per-model queries
    per_strip_per_model: List[Dict[str, Optional[Dict[str, Any]]]] = [
        {} for _ in strips
    ]
    raw_responses: List[Dict[str, Dict[str, Any]]] = [
        {} for _ in strips
    ]
    api_success_count = 0

    if skip_api or not chosen_models:
        print("[decipher] skip_api=True or empty model list — skipping OpenRouter calls")
    else:
        client = OpenRouterClient(api_key=api_key)
        for model in chosen_models:
            print(f"[decipher]   querying {model.slug} on {len(strips)} strips ...")
            t0 = time.time()
            tasks = []
            for s in strips:
                _, hint_block = hints_by_strip.get(s.index, ([], ""))
                ctx = StripPromptContext(
                    seg_id=seg_id, strip_index=s.index, y_center=s.y_center,
                    hint_block=hint_block,
                )
                tasks.append(_query_model_for_strip(
                    client, model, s, ctx,
                    image_bytes=plate_bytes_by_strip.get(s.index),
                ))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            dt = time.time() - t0
            for s, r in zip(strips, results):
                if isinstance(r, Exception):
                    r = CompletionResult(
                        model=model.slug,
                        text="",
                        parsed=None,
                        finish_reason=None,
                        usage=None,
                        error=repr(r),
                    )
                per_strip_per_model[s.index][model.slug] = r.parsed
                raw_responses[s.index][model.slug] = {
                    "tier": model.tier,
                    "display_name": model.display_name,
                    "raw": r.text,
                    "parsed": r.parsed,
                    "error": r.error,
                    "finish_reason": r.finish_reason,
                    "usage": r.usage,
                }
            ok = sum(1 for r in results if r.error is None)
            api_success_count += ok
            print(f"    done in {dt:.1f}s  ({ok}/{len(strips)} ok)")

    existing_result = out_dir / "result.json"
    if not skip_api and chosen_models and api_success_count == 0 and existing_result.exists():
        print(
            "[decipher] all API calls failed; preserving existing result.json "
            "instead of overwriting it with local fallback"
        )
        return json.loads(existing_result.read_text(encoding="utf-8"))

    # ---- merge manual readings
    manual_dir = out_dir / "manual"
    manual_by_slug = _load_manual_readings(manual_dir)
    for slug, by_idx in manual_by_slug.items():
        spec = next((m for m in MANUAL_MODELS if m.slug == slug), None)
        for s in strips:
            parsed = by_idx.get(s.index)
            per_strip_per_model[s.index][slug] = parsed
            raw_responses[s.index][slug] = {
                "tier": spec.tier if spec else 0,
                "display_name": spec.display_name if spec else slug,
                "raw": json.dumps(parsed) if parsed else "",
                "parsed": parsed,
                "error": None if parsed else "no entry",
                "finish_reason": "manual",
                "usage": None,
            }

    # ---- consensus per strip
    strips_out: List[Dict[str, Any]] = []
    for s in strips:
        consensus = build_consensus(per_strip_per_model[s.index])
        hint_records, _ = hints_by_strip.get(s.index, ([], ""))
        consensus_dict = consensus.to_dict()
        if not consensus_dict["characters"] and hint_records:
            consensus_dict = _consensus_from_template_hints(hint_records)
            raw_responses[s.index]["local/template_hints"] = {
                "tier": 9,
                "display_name": "Local template hints",
                "raw": json.dumps(consensus_dict, ensure_ascii=False),
                "parsed": {
                    "line_text": consensus_dict["text"],
                    "characters": consensus_dict["characters"],
                    "translation_en": "[uncertain]",
                    "probable_summary": "[uncertain]",
                    "notes": consensus_dict["notes"],
                    "overall_confidence": "low",
                },
                "error": None,
                "finish_reason": "fallback",
                "usage": None,
            }
        strips_out.append({
            "strip_id": s.index,
            "image_path": f"strips/strip_{s.index:02d}.png",
            "reading_plate_path": f"plates/plate_{s.index:02d}.png",
            "y_range": [s.y0, s.y1],
            "x_range": [s.x0, s.x1],
            "y_center": s.y_center,
            "template_hints": hint_records,
            "per_model": raw_responses[s.index],
            "consensus": consensus_dict,
        })

    segment_summary = _segment_summary(strips_out)

    result = {
        "seg_id": seg_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "label_path": str(label_path),
        "models_used_open_source": [m.slug for m in chosen_models] if not skip_api else [],
        "models_used_manual": list(manual_by_slug.keys()),
        "models_used_local": ["local/template_hints"],
        "n_strips": len(strips),
        "segment_summary": segment_summary,
        "segment_text": segment_summary["text"],
        "segment_meaning": segment_summary["probable_summary"],
        "segment_translation_en": segment_summary["english_translation"],
        "probable_scroll_summary": segment_summary["probable_scroll_summary"],
        "strips": strips_out,
    }
    detection_path = out_dir / "result.detection.json"
    result = align_result_characters(
        result,
        strips_dir=strips_dir,
        detection_lines=load_detection_lines(detection_path),
    )
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[decipher] wrote {out_dir / 'result.json'}")
    return result


def decipher_segment(**kw) -> Dict[str, Any]:
    """Synchronous wrapper."""
    return asyncio.run(decipher_segment_async(**kw))
