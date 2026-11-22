"""CLI Training Script for DermaAI Model Architectures."""

import argparse
from pathlib import Path
from src.config import load_config
from src.utils import set_seed, setup_logger
from src.models.baseline_cnn import build_baseline_cnn
from src.models.transfer_models import build_transfer_model
from src.models.hybrid_model import build_hybrid_attention_model

logger = setup_logger("DermaAI.TrainScript")


def parse_args():
    parser = argparse.ArgumentParser(description="Train skin lesion classification models.")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_cnn",
        choices=["baseline_cnn", "resnet50", "efficientnet_b0", "mobilenet_v2", "attention_cbam"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="focal_loss", help="Loss function")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    set_seed(config.project.seed)

    logger.info(f"Initializing training pipeline for model: {args.model}")
    target_size = tuple(config.data.image_size)

    if args.model == "baseline_cnn":
        model = build_baseline_cnn(input_shape=(*target_size, 3), num_classes=config.data.num_classes)
    elif args.model == "attention_cbam":
        model = build_hybrid_attention_model(input_shape=(*target_size, 3), num_classes=config.data.num_classes)
    else:
        model = build_transfer_model(
            backbone_name=args.model,
            input_shape=(*target_size, 3),
            num_classes=config.data.num_classes,
        )

    model.summary(print_fn=logger.info)
    logger.info(f"Model architecture '{args.model}' configured successfully.")


if __name__ == "__main__":
    main()
