"""CLI Script to Generate Complete Dataset Visualization and Markdown Audit Report."""

import argparse
from pathlib import Path
import pandas as pd

from src.config import load_config
from src.data.metadata_parser import ClinicalMetadataParser
from src.evaluation.dataset_reporter import DatasetReporter
from src.utils import setup_logger

logger = setup_logger("DermaAI.GenerateReport")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual charts and dataset audit reports.")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save visual charts and report")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    metadata_path = Path(args.metadata) if args.metadata else Path(config.paths.metadata_csv)

    if not metadata_path.exists():
        logger.warning(
            f"Metadata not found at '{metadata_path}'. Generating report requires metadata. "
            "Run 'python scripts/create_sample_dataset.py' to generate test data."
        )
        return

    logger.info(f"Loading metadata from {metadata_path}...")
    parser = ClinicalMetadataParser(metadata_path)
    df = parser.load_metadata()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reporter = DatasetReporter(df)
    reporter.plot_class_distribution(out_dir / "class_distribution.png")
    reporter.plot_demographics(out_dir / "demographics_distribution.png")
    reporter.generate_markdown_report(out_dir / "dataset_analysis_report.md")

    logger.info("Dataset visualization reports generated successfully.")


if __name__ == "__main__":
    main()
