"""Command‑line demo for DermaAI XAI.

Usage example::

    python scripts/run_xai_demo.py \
        --model reports/checkpoints/best_resnet50.keras \
        --image dataset/raw/sample_images/ISIC_SAMPLE_00001.jpg \
        --output_dir reports/xai_demo

The script loads the model, runs Grad‑CAM, saves the original image, heatmap,
overlay and a JSON manifest, and also produces a multi‑panel figure showing the
prediction probabilities.
"""

import argparse
import pathlib

from src.explainability.explainer import DermaExplainer
from src.explainability.visualization import plot_prediction_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single‑image XAI explanation for DermaAI.")
    parser.add_argument("--model", required=True, help="Path to a Keras checkpoint (.keras or .h5).")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument(
        "--output_dir",
        default="reports/xai_demo",
        help="Directory where explanation artefacts will be stored.",
    )
    parser.add_argument(
        "--class_names",
        nargs="*",
        default=None,
        help="Optional list of class names matching the model output order.",
    )
    args = parser.parse_args()

    explainer = DermaExplainer(str(args.model), class_names=args.class_names)
    result = explainer.explain(str(args.image))
    explainer.save_explanation(result, args.output_dir)

    # Also generate a nice figure for quick visual inspection
    fig_path = pathlib.Path(args.output_dir) / pathlib.Path(args.image).stem / "panel.png"
    plot_prediction_panel(
        image=result.overlay,  # reuse the overlay as the base image for figure simplicity
        heatmap=result.heatmap,
        overlay=result.overlay,
        class_probs=result.class_probs,
        class_names=args.class_names,
        predicted_class=result.class_idx,
        save_path=str(fig_path),
    )
    print(f"Explanation saved to {args.output_dir}")


if __name__ == "__main__":
    main()
