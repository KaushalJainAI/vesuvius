"""Summarize standardized model runs into one comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "models" / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(args.runs_root.iterdir() if args.runs_root.exists() else []):
        if not run_dir.is_dir():
            continue
        hist_path = run_dir / "history.json"
        cfg_path = run_dir / "config.json"
        if not hist_path.exists():
            continue
        history = json.loads(hist_path.read_text())
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if not history:
            continue
        best = max(history, key=lambda e: e.get("f0.5", -1))
        rows.append({
            "run": run_dir.name,
            "model": cfg.get("model", "?"),
            "labels": "refined" if cfg.get("label_root") else "original",
            "val": cfg.get("val_segment", "?"),
            "epoch": best.get("epoch"),
            "f0.5": best.get("f0.5"),
            "precision": best.get("precision"),
            "recall": best.get("recall"),
        })

    if not rows:
        print("No standardized runs found.")
        return

    header = f"{'run':<34} {'model':<10} {'labels':<9} {'val':<14} {'ep':>3} {'F0.5':>7} {'P':>7} {'R':>7}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: r["f0.5"], reverse=True):
        print(
            f"{row['run']:<34} {row['model']:<10} {row['labels']:<9} {row['val']:<14} "
            f"{row['epoch']:>3} {row['f0.5']:>7.4f} {row['precision']:>7.4f} {row['recall']:>7.4f}"
        )


if __name__ == "__main__":
    main()

