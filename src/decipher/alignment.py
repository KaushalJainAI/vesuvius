"""Map recognized strip characters back onto the full ink-label image.

The model output is text-first, while the ink-map UI is geometry-first. This
module adds the missing bridge: approximate character boxes in strip pixels,
full-segment pixels, and nearest detection-line/blob metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage import measure, morphology


def _bbox_xywh_to_xyxy(box: List[int]) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in box[:4]]
    return x, y, x + w, y + h


def _overlap_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    w = max(0, min(ax1, bx1) - max(ax0, bx0))
    h = max(0, min(ay1, by1) - max(ay0, by0))
    return w * h


def _components_for_strip(strip_u8: np.ndarray, threshold: float = 0.55) -> List[Dict[str, Any]]:
    prob = strip_u8.astype(np.float32) / 255.0
    binary = prob > threshold
    binary = morphology.remove_small_objects(binary, min_size=80)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    lab = measure.label(binary)
    comps: List[Dict[str, Any]] = []
    for p in measure.regionprops(lab, intensity_image=prob):
        if p.area < 120:
            continue
        y0, x0, y1, x1 = p.bbox
        comps.append({
            "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "xyxy": (int(x0), int(y0), int(x1), int(y1)),
            "cx": float(p.centroid[1]),
            "cy": float(p.centroid[0]),
            "area": int(p.area),
            "mean_prob": float(p.mean_intensity),
        })
    return comps


def _estimate_char_box(
    *,
    char: Dict[str, Any],
    strip_w: int,
    strip_h: int,
    comps: List[Dict[str, Any]],
    prev_x: Optional[float],
    next_x: Optional[float],
) -> Dict[str, Any]:
    x_norm = max(0.0, min(1.0, float(char.get("x_norm", 0.5))))
    cx = x_norm * strip_w
    spacing_left = abs(cx - prev_x) if prev_x is not None else strip_w * 0.08
    spacing_right = abs(next_x - cx) if next_x is not None else strip_w * 0.08
    expected_w = int(max(70, min(260, min(spacing_left, spacing_right) * 0.9)))
    expected_h = int(max(60, min(strip_h, strip_h * 0.72)))
    fallback = (
        int(max(0, cx - expected_w / 2)),
        int(max(0, strip_h / 2 - expected_h / 2)),
        int(min(strip_w, cx + expected_w / 2)),
        int(min(strip_h, strip_h / 2 + expected_h / 2)),
    )

    # Prefer visible ink components whose centre is near the predicted x.
    max_dx = max(expected_w * 0.9, 90)
    nearby = [c for c in comps if abs(c["cx"] - cx) <= max_dx]
    if nearby:
        nearby.sort(key=lambda c: (abs(c["cx"] - cx), -c["area"]))
        chosen = nearby[:3]
        x0 = min(c["xyxy"][0] for c in chosen)
        y0 = min(c["xyxy"][1] for c in chosen)
        x1 = max(c["xyxy"][2] for c in chosen)
        y1 = max(c["xyxy"][3] for c in chosen)
        pad_x = max(18, int((x1 - x0) * 0.35))
        pad_y = max(12, int((y1 - y0) * 0.22))
        box = (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(strip_w, x1 + pad_x),
            min(strip_h, y1 + pad_y),
        )
        source = "component_snap"
        component_count = len(chosen)
    else:
        box = fallback
        source = "estimated"
        component_count = 0

    x0, y0, x1, y1 = box
    return {
        "bbox_strip": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "alignment_source": source,
        "component_count": component_count,
    }


def _line_for_box(full_box: List[int], detection_lines: List[Dict[str, Any]]) -> Tuple[Optional[int], List[int]]:
    if not detection_lines:
        return None, []
    x, y, w, h = [int(v) for v in full_box]
    box_xyxy = (x, y, x + w, y + h)
    cy = y + h / 2.0

    best_line: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    for line in detection_lines:
        y0, y1 = line.get("y_band", [0, 0])
        ly = (float(y0) + float(y1)) / 2.0
        inside_penalty = 0 if y0 <= cy <= y1 else min(abs(cy - y0), abs(cy - y1))
        score = inside_penalty + abs(cy - ly) * 0.05
        if score < best_score:
            best_score = score
            best_line = line

    if best_line is None:
        return None, []

    linked: List[int] = []
    for i, blob in enumerate(best_line.get("blob_bboxes", []) or []):
        if _overlap_area(box_xyxy, _bbox_xywh_to_xyxy(blob)) > 0:
            linked.append(i)
    return int(best_line.get("line_no", -1)), linked


def load_detection_lines(detection_path: Path) -> List[Dict[str, Any]]:
    if not detection_path.exists():
        return []
    try:
        data = json.loads(detection_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    lines = data.get("lines")
    return lines if isinstance(lines, list) else []


def align_result_characters(
    result: Dict[str, Any],
    *,
    strips_dir: Path,
    detection_lines: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Mutate and return ``result`` with per-character full-image boxes."""
    detection_lines = detection_lines or []
    all_boxes: List[Dict[str, Any]] = []

    for strip in result.get("strips", []) or []:
        consensus = strip.get("consensus") or {}
        chars = consensus.get("characters") or []
        if not chars:
            strip["line_transcription"] = ""
            continue

        image_path = strip.get("image_path") or f"strips/strip_{int(strip.get('strip_id', 0)):02d}.png"
        png_path = strips_dir.parent / image_path
        if not png_path.exists():
            png_path = strips_dir / Path(image_path).name
        try:
            from PIL import Image
            strip_u8 = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8)
        except Exception:
            continue

        strip_h, strip_w = strip_u8.shape
        comps = _components_for_strip(strip_u8)
        x0_full, y0_full = int(strip.get("x_range", [0, 0])[0]), int(strip.get("y_range", [0, 0])[0])
        xs = [max(0.0, min(1.0, float(c.get("x_norm", 0.5)))) * strip_w for c in chars]

        line_chars: Dict[int, List[str]] = {}
        for i, char in enumerate(chars):
            prev_x = xs[i - 1] if i > 0 else None
            next_x = xs[i + 1] if i + 1 < len(xs) else None
            aligned = _estimate_char_box(
                char=char, strip_w=strip_w, strip_h=strip_h, comps=comps,
                prev_x=prev_x, next_x=next_x,
            )
            sx, sy, sw, sh = aligned["bbox_strip"]
            full_box = [x0_full + sx, y0_full + sy, sw, sh]
            line_no, linked = _line_for_box(full_box, detection_lines)

            char["bbox_strip"] = [sx, sy, sw, sh]
            char["bbox_full"] = full_box
            char["line_no"] = line_no
            char["linked_blob_indices"] = linked
            char["alignment_source"] = aligned["alignment_source"]
            char["component_count"] = aligned["component_count"]

            rec = {
                "strip_id": strip.get("strip_id"),
                "char_index": i,
                "char": char.get("char"),
                "confidence": char.get("confidence"),
                "tier": char.get("tier"),
                "bbox_full": full_box,
                "line_no": line_no,
                "linked_blob_indices": linked,
                "alignment_source": char["alignment_source"],
            }
            all_boxes.append(rec)
            if line_no is not None and line_no >= 0:
                line_chars.setdefault(line_no, []).append(str(char.get("char", "")))

        strip["line_transcription"] = consensus.get("text") or "".join(str(c.get("char", "")) for c in chars)

    result["character_boxes"] = all_boxes
    result["alignment_summary"] = {
        "n_character_boxes": len(all_boxes),
        "n_component_snapped": sum(1 for b in all_boxes if b.get("alignment_source") == "component_snap"),
        "n_linked_to_blobs": sum(1 for b in all_boxes if b.get("linked_blob_indices")),
    }
    return result
