"""
src/refine_loop.py

Orchestrator for the iterative label refinement loop.

Two modes:
  label_only — filters + LLM only, no GPU retraining (~1-2 hrs/segment/iteration)
  full_loop  — includes SegmentInkNet retraining on refined labels (~8-12 hrs/cycle)

Main entry point:
  run_refinement(config) → (final_labels, iteration_history)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from label_blend import (
    ALPHA_SCHEDULE,
    blend_labels,
    compute_diff_stats,
    log_iteration,
    save_iteration_labels,
)
from label_filters import apply_filters, load_label
from letter_candidates import (
    GreekTemplateMatcher,
    assign_confidence_tier,
    detect_text_lines,
    extract_candidates,
)
from llm_gap_fill import LLMGapFiller


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RefinementConfig:
    seg_id: str
    """Segment ID, e.g. '20231221180251'."""

    label_path: Path
    """Starting label map (.npy or .tif). Usually data/labelled_segments/{id}/ink_labels.tif."""

    mode: str = "label_only"
    """'label_only' — filters + LLM, no retraining.
       'full_loop'  — includes SegmentInkNet retraining after each iteration."""

    max_iterations: int = 4
    alpha_schedule: List[float] = field(default_factory=lambda: list(ALPHA_SCHEDULE))

    # Candidate extraction
    threshold: float = 0.40
    min_area: int = 50
    max_area: int = 8000
    top_k_matches: int = 3

    # Filter settings
    median_radius: int = 1
    clahe_clip: float = 0.02

    # LLM settings
    llm_model: str = "claude-opus-4-7"
    llm_batch_size: int = 5

    # Stopping criterion: stop when mean |Δ| < this between iterations
    convergence_threshold: float = 0.01

    # Output paths
    out_dir: Path = Path("predictions/refinement")
    log_path: Path = Path("models/refinement_log.json")

    # full_loop only: path to a Python training script that accepts
    # --seg-id and --label-version flags
    train_script: Optional[str] = None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_refinement(
    config: RefinementConfig,
    *,
    api_key: Optional[str] = None,
    matcher: Optional[GreekTemplateMatcher] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Run the label refinement loop.

    Each iteration:
      Stage 1: apply_filters          — enhance contrast, repair strokes
      Stage 2: extract + match        — letter candidates + Greek template matching
      Stage 3: LLM gap-fill           — Claude resolves ambiguous + gap positions
      Stage 4: blend                  — α × llm_map + (1-α) × prev_labels
      [Stage 5: retrain if full_loop] — SegmentInkNet trained on blended labels

    Returns:
      (final_labels, history) — float32 refined label map, per-iteration stats.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    config.out_dir = Path(config.out_dir)
    config.log_path = Path(config.log_path)
    config.label_path = Path(config.label_path)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_label(config.label_path)
    H, W = labels.shape
    _log(f"[refine] Starting — segment {config.seg_id}  labels {labels.shape}  mode={config.mode}")

    if matcher is None:
        _log("[refine] Building Greek template matcher (one-time)…")
        matcher = GreekTemplateMatcher()
        _log("  Done.")

    filler = LLMGapFiller(api_key=api_key, model=config.llm_model)
    history: List[Dict] = []
    prev_labels = labels.copy()

    for iteration in range(1, config.max_iterations + 1):
        alpha = config.alpha_schedule[min(iteration - 1, len(config.alpha_schedule) - 1)]
        _log(f"\n{'=' * 64}")
        _log(f"[refine] Iteration {iteration}/{config.max_iterations}   α={alpha:.2f}")
        _log(f"{'=' * 64}")

        # ── Stage 1: Filter pipeline ──────────────────────────────────────
        _log("[1/4] Filter pipeline…")
        filtered = apply_filters(
            labels,
            median_radius=config.median_radius,
            clahe_clip=config.clahe_clip,
            threshold=config.threshold,
        )

        # ── Stage 2: Candidates + template matching ───────────────────────
        _log("[2/4] Extracting letter candidates + template matching…")
        candidates = extract_candidates(
            filtered,
            threshold=config.threshold,
            min_area=config.min_area,
            max_area=config.max_area,
        )
        _log(f"      {len(candidates)} candidates found")

        for cand in candidates:
            if assign_confidence_tier(cand) != "LOW":
                cand.matches = matcher.match(cand.patch, top_k=config.top_k_matches)

        lines = detect_text_lines(candidates)
        _log(f"      {len(lines)} text lines detected")

        # ── Stage 3: LLM gap-filling ──────────────────────────────────────
        _log("[3/4] LLM gap-filling…")
        fill_results = filler.fill_lines(
            lines, W, H, batch_size=config.llm_batch_size
        )
        updated = filler.apply_to_prob_map(filtered, fill_results)

        # ── Stage 4: Blend ─────────────────────────────────────────────────
        _log(f"[4/4] Blending labels (α={alpha})…")
        blended = blend_labels(labels, updated, alpha=alpha)

        # Convergence stats
        stats = compute_diff_stats(prev_labels, blended)
        entry = {
            "iteration":             iteration,
            "seg_id":                config.seg_id,
            "mode":                  config.mode,
            "alpha":                 alpha,
            "n_candidates":          len(candidates),
            "n_lines":               len(lines),
            "n_llm_results":         len(fill_results),
            **stats,
        }
        history.append(entry)
        log_iteration(entry, config.log_path)

        _log(f"      mean|Δ|={stats['mean_abs_diff']:.4f}  "
             f"+{stats['pixels_increased_pct']:.1f}%  "
             f"-{stats['pixels_decreased_pct']:.1f}%")

        # Save iteration output
        npy_p, tif_p = save_iteration_labels(blended, config.seg_id, iteration, config.out_dir)
        _log(f"      Saved → {tif_p.name}")

        # ── Stage 5 (full_loop): retrain model ────────────────────────────
        if config.mode == "full_loop":
            _run_training(config, iteration, blended, _log)

        prev_labels = labels.copy()
        labels = blended

        if stats["mean_abs_diff"] < config.convergence_threshold:
            _log(f"\n[refine] Converged at iteration {iteration} "
                 f"(mean|Δ| {stats['mean_abs_diff']:.4f} < {config.convergence_threshold})")
            break

    _log(f"\n[refine] Finished. Final labels in {config.out_dir}")
    return labels, history


# ---------------------------------------------------------------------------
# Training hook (full_loop mode)
# ---------------------------------------------------------------------------

def _run_training(
    config: RefinementConfig,
    iteration: int,
    refined_labels: np.ndarray,
    log,
) -> None:
    import subprocess
    import tifffile as tf

    label_dst = (
        Path("data/labelled_segments")
        / config.seg_id
        / f"ink_labels_v{iteration}.tif"
    )
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    tf.imwrite(str(label_dst), (refined_labels * 255).astype(np.uint8))
    log(f"  [train] Wrote label → {label_dst}")

    if not config.train_script:
        log("  [train] No train_script configured — skipping model retraining.")
        return

    cmd = [
        sys.executable, config.train_script,
        "--seg-id", config.seg_id,
        "--label-version", str(iteration),
    ]
    log(f"  [train] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  [train] WARNING: script exited {result.returncode}")
        if result.stderr:
            log(result.stderr[-600:])
    else:
        log("  [train] Complete.")
