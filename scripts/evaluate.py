"""CLI Evaluation Script for DermaAI Models."""

import argparse
from pathlib import Path
from src.config import load_config
from src.utils import setup_logger
from src.evaluation.evaluator import ModelEvaluator

logger = setup_logger("DermaAI.EvalScript")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DermaAI model checkpoints.")
    parser.add_argument("--weights", type=str, required=False, help="Path to saved model weights")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save evaluation reports")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    class_names = [cls_cfg["code"] for cls_cfg in config.data.classes]
    evaluator = ModelEvaluator(class_names=class_names)
    logger.info(f"Evaluator initialized for classes: {class_names}")


if __name__ == "__main__":
    main()
