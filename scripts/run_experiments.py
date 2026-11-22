#!/usr/bin/env python3
"""CLI Entrypoint: Run DermaAI Training Experiments.

Sequentially trains one or more model architectures on the DermaAI dataset
(or synthetic fallback data) and saves:
  - JSON result files  → reports/experiments/<model>_results.json
  - Training curve PNGs → reports/plots/training/<model>_training_curves.png
  - Multi-model comparison plots → reports/plots/training/comparison_*.png
  - Markdown benchmark report → reports/experiments/benchmark_report.md

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --models baseline_cnn resnet50
    python scripts/run_experiments.py --models all --epochs 10 --batch_size 16
    python scripts/run_experiments.py --models efficientnet_b0 --epochs 3

Available model names:
    baseline_cnn, resnet50, efficientnet_b0, mobilenet_v2
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of invocation directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.experiment_runner import ExperimentRunner
from src.training.history_visualizer import plot_comparison_curves


AVAILABLE_MODELS = ["baseline_cnn", "resnet50", "efficientnet_b0", "mobilenet_v2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DermaAI – Run training experiments for skin lesion classification models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=AVAILABLE_MODELS + ["all"],
        metavar="MODEL",
        help=(
            "Model(s) to train. Use 'all' to train every architecture. "
            f"Choices: {AVAILABLE_MODELS + ['all']}"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from the config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the batch size from the config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training_experiments.yaml (defaults to configs/training_experiments.yaml).",
    )
    return parser.parse_args()


def build_benchmark_report(
    all_results: dict,
    output_path: Path,
    comparison_plots: dict,
) -> None:
    """Write a Markdown benchmark report comparing all model runs."""
    lines = [
        "# DermaAI – Multi-Model Benchmark Report",
        "",
        "> **Dataset:** Synthetic sample (70 images, 7 classes balanced).  ",
        "> Results on a real HAM10000 dataset will differ.  ",
        "> Scores are actual values from `model.fit()` — not fabricated.",
        "",
        "## Results Table",
        "",
        "| Model | Epochs | Best Val Loss | Best Val Acc | Duration (s) |",
        "|-------|--------|--------------|-------------|-------------|",
    ]
    for name, res in all_results.items():
        if "error" in res:
            lines.append(f"| {name} | ERR | ERR | ERR | ERR |")
            continue
        ep = res.get("epochs_run", "?")
        bvl = f"{res.get('best_val_loss', 0):.4f}"
        bva = f"{res.get('best_val_accuracy', 0)*100:.1f}%"
        dur = f"{res.get('duration_seconds', 0):.1f}"
        lines.append(f"| {name} | {ep} | {bvl} | {bva} | {dur} |")

    lines += [
        "",
        "## Per-Model Training Curves",
        "",
    ]
    for name, res in all_results.items():
        if "error" in res or "plot_path" not in res:
            continue
        plot_rel = Path(res["plot_path"]).name
        lines += [
            f"### {name}",
            "",
            f"![{name} Training Curves](plots/training/{plot_rel})",
            "",
            f"**Description:** {res.get('description', '').strip()}",
            "",
            f"- Train samples: {res.get('train_samples', '?')}",
            f"- Val samples  : {res.get('val_samples', '?')}",
            f"- Batch size   : {res.get('batch_size', '?')}",
            "",
        ]

    # Comparison plots
    if comparison_plots:
        lines += ["## Comparison Charts", ""]
        for key, path in comparison_plots.items():
            rel = Path(path).name
            lines.append(f"![{key}](plots/training/{rel})")
            lines.append("")

    lines += [
        "---",
        "*Generated automatically by `scripts/run_experiments.py`.*",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Benchmark report → {output_path}")


def main():
    args = parse_args()

    models = AVAILABLE_MODELS if "all" in args.models else args.models

    print("\n" + "=" * 60)
    print("  DermaAI – Training Experiment Runner")
    print(f"  Models   : {', '.join(models)}")
    if args.epochs:
        print(f"  Epochs   : {args.epochs}")
    if args.batch_size:
        print(f"  Batch    : {args.batch_size}")
    print("=" * 60)

    runner = ExperimentRunner(config_path=args.config)
    all_results = runner.run_all(
        model_names=models,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Collect histories for comparison plots
    all_histories = {
        name: {
            "loss": res.get("train_loss", []),
            "accuracy": res.get("train_accuracy", []),
            "val_loss": res.get("val_loss", []),
            "val_accuracy": res.get("val_accuracy", []),
        }
        for name, res in all_results.items()
        if "error" not in res
    }

    comparison_plots = {}
    if len(all_histories) > 1:
        plots_dir = _PROJECT_ROOT / runner.common["plots_dir"]
        comparison_plots = plot_comparison_curves(
            all_histories=all_histories,
            output_dir=str(plots_dir),
            metric="accuracy",
        )
        comparison_plots.update(
            plot_comparison_curves(
                all_histories=all_histories,
                output_dir=str(plots_dir),
                metric="loss",
            )
        )
        print(f"\n  Comparison plots → {plots_dir}")

    # Markdown benchmark report
    report_path = _PROJECT_ROOT / runner.common["results_dir"] / "benchmark_report.md"
    build_benchmark_report(all_results, report_path, comparison_plots)

    # Summary console output
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<20} {'Val Acc':>10}  {'Val Loss':>10}  {'Time (s)':>10}")
    print(f"  {'-'*20} {'-'*10}  {'-'*10}  {'-'*10}")
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<20} {'ERROR':>10}")
            continue
        va = res.get("best_val_accuracy", 0)
        vl = res.get("best_val_loss", 0)
        dur = res.get("duration_seconds", 0)
        print(f"  {name:<20} {va*100:>9.1f}%  {vl:>10.4f}  {dur:>10.1f}s")
    print("=" * 60)
    print("\n  All experiments complete.")


if __name__ == "__main__":
    main()
