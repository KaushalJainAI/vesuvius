"""Run the decipher pipeline against the readable Kaggle-fragment label image.

This is a SANITY CHECK: feed the LLMs a label image that humans CAN read
(the high-quality manual ink annotation at web/public/assets/samples/
ink_labels_real.png) and observe what they produce. This proves the
pipeline works end-to-end and isolates the upstream resolution problem
as the real bottleneck.

Output: predictions/decipher/sample_readable/result.json
        + mirrored to web/public/assets/decipher/sample_readable/

The webapp picks it up automatically because the orchestrator's
write_segments_index() walks every dir under web/public/assets/decipher/.
"""
from __future__ import annotations

import asyncio
import io
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decipher.consensus import build_consensus  # noqa: E402
from decipher.model_registry import OPEN_SOURCE_MODELS, MANUAL_MODELS  # noqa: E402
from decipher.openrouter_client import OpenRouterClient  # noqa: E402
from decipher.prompt_builder import (  # noqa: E402
    SYSTEM_PROMPT, StripPromptContext, build_user_prompt,
)
from decipher.strip_extractor import Strip  # noqa: E402
from decipher.template_hints import build_prompt_hints  # noqa: E402


SAMPLE_PATH = ROOT / "web" / "public" / "assets" / "samples" / "ink_labels_real.png"
SEG_ID = "sample_readable_kaggle"
OUT_DIR = ROOT / "predictions" / "decipher" / SEG_ID
WEB_DIR = ROOT / "web" / "public" / "assets" / "decipher" / SEG_ID

UPSCALE = 3   # 512×388 → 1536×1164 — better for vision encoders


def make_whole_image_strip(img: np.ndarray) -> Strip:
    """Send the WHOLE image as a single strip, upscaled. Models read all lines."""
    H, W = img.shape
    up = Image.fromarray(img, mode="L").resize(
        (W * UPSCALE, H * UPSCALE), Image.BICUBIC,
    )
    buf = io.BytesIO()
    up.save(buf, format="PNG", optimize=True)
    return Strip(
        index=0, y0=0, y1=H, x0=0, x1=W,
        image_u8=img, png_bytes=buf.getvalue(),
    )


async def run() -> Dict[str, Any]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(SAMPLE_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "strips").mkdir(parents=True, exist_ok=True)

    print(f"[sample] loading {SAMPLE_PATH}")
    img = np.asarray(Image.open(SAMPLE_PATH).convert("L"))
    print(f"  -> {img.shape} (small + readable, NOT a smoothed blob map)")

    strips = [make_whole_image_strip(img)]
    for s in strips:
        (OUT_DIR / "strips" / f"strip_{s.index:02d}.png").write_bytes(s.png_bytes)
    print(f"  -> sending WHOLE image (1 strip, upscaled {UPSCALE}x)")

    # Build template hints from each strip
    print(f"[sample] building template hints ...")
    hints_by_strip: Dict[int, tuple] = {}
    for s in strips:
        try:
            records, block = build_prompt_hints(s.png_bytes)
            hints_by_strip[s.index] = (records, block)
            print(f"  strip {s.index}: {len(records)} template candidates")
        except Exception as e:
            print(f"  strip {s.index}: hint build failed ({e!r})")
            hints_by_strip[s.index] = ([], "")

    # Query all open-source models
    client = OpenRouterClient()
    per_strip_per_model: List[Dict[str, Any]] = [{} for _ in strips]
    raw_responses: List[Dict[str, Any]] = [{} for _ in strips]

    for model in OPEN_SOURCE_MODELS:
        print(f"[sample]   querying {model.slug} on {len(strips)} strips ...")
        tasks = []
        for s in strips:
            _, hint_block = hints_by_strip[s.index]
            ctx = StripPromptContext(
                seg_id=SEG_ID, strip_index=s.index, y_center=s.y_center,
                hint_block=hint_block,
            )
            tasks.append(client.chat_with_image(
                model=model.slug,
                system_prompt=SYSTEM_PROMPT,
                user_text=build_user_prompt(ctx),
                image_bytes=s.png_bytes,
                image_mime="image/png",
            ))
        results = await asyncio.gather(*tasks, return_exceptions=False)
        ok = sum(1 for r in results if r.error is None and r.parsed)
        print(f"    {ok}/{len(strips)} returned parsable JSON")
        for s, r in zip(strips, results):
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

    # Build per-strip consensus
    strips_out = []
    for s in strips:
        hint_records, _ = hints_by_strip[s.index]
        cons = build_consensus(per_strip_per_model[s.index])
        strips_out.append({
            "strip_id": s.index,
            "image_path": f"strips/strip_{s.index:02d}.png",
            "y_range": [s.y0, s.y1],
            "x_range": [s.x0, s.x1],
            "y_center": s.y_center,
            "template_hints": hint_records,
            "per_model": raw_responses[s.index],
            "consensus": cons.to_dict(),
        })

    result = {
        "seg_id": SEG_ID,
        "created": "real",
        "source_note": (
            "Kaggle competition fragment with manual human-annotated ink labels. "
            "Used as a sanity check to demonstrate the decipher pipeline against "
            "ACTUALLY READABLE Greek text, in contrast to the smoothed CNN output "
            "of the PHercParis4 segments."
        ),
        "label_path": str(SAMPLE_PATH),
        "models_used_open_source": [m.slug for m in OPEN_SOURCE_MODELS],
        "models_used_manual": [],
        "n_strips": len(strips),
        "strips": strips_out,
    }
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[sample] wrote {OUT_DIR / 'result.json'}")

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_DIR / "result.json", WEB_DIR / "result.json")
    web_strips = WEB_DIR / "strips"
    if web_strips.exists():
        shutil.rmtree(web_strips)
    shutil.copytree(OUT_DIR / "strips", web_strips)
    print(f"[sample] mirrored to {WEB_DIR}")

    # Update index.json
    index_path = ROOT / "web" / "public" / "assets" / "decipher" / "index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        idx = {"segments": []}
    if not any(e["seg_id"] == SEG_ID for e in idx["segments"]):
        idx["segments"].append({"seg_id": SEG_ID, "n_strips": len(strips), "mock": False})
        index_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        print(f"[sample] updated index.json")
    return result


if __name__ == "__main__":
    asyncio.run(run())
