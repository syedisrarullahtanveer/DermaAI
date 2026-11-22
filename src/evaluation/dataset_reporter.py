"""Dataset Visualization and Markdown Clinical Audit Report Generator."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.dataset_analyzer import DatasetDistributionAnalyzer
from src.utils import setup_logger

logger = setup_logger("DermaAI.DatasetReporter")


class DatasetReporter:
    """Generates visual charts and comprehensive Markdown audit reports for skin lesion datasets."""

    def __init__(self, metadata_df: pd.DataFrame, class_col: str = "dx"):
        self.df = metadata_df.copy()
        self.class_col = class_col
        self.analyzer = DatasetDistributionAnalyzer(self.df, class_col=class_col)

    def plot_class_distribution(self, output_path: Union[str, Path]) -> None:
        """Render high-resolution class distribution bar chart with clinical risk coloration."""
        counts = self.df[self.class_col].value_counts()
        total = len(self.df)

        risk_color_map = {
            "mel": "#ef4444",   # Red (Critical)
            "bcc": "#f97316",   # Orange (High)
            "akiec": "#eab308", # Amber (High)
            "bkl": "#3b82f6",   # Blue (Low)
            "nv": "#06b6d4",    # Cyan (Low)
            "df": "#8b5cf6",    # Purple (Low)
            "vasc": "#10b981",  # Emerald (Low)
        }
        colors = [risk_color_map.get(cls, "#64748b") for cls in counts.index]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="#1e293b", linewidth=1.2)

        for bar, count in zip(bars, counts.values):
            pct = (count / total) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(counts.values) * 0.015),
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.title("DermaAI: Lesion Class Frequency & Imbalance Distribution", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Diagnosis Code", fontsize=11, labelpad=8)
        plt.ylabel("Sample Count", fontsize=11, labelpad=8)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        plt.close()
        logger.info(f"Class distribution chart saved to {out}")

    def plot_demographics(self, output_path: Union[str, Path]) -> None:
        """Render multi-panel demographic distributions (Age Histogram, Anatomical Site)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Age Histogram
        if "age" in self.df.columns:
            valid_age = self.df["age"].dropna().astype(float)
            sns.histplot(valid_age, kde=True, ax=axes[0], color="#2563eb", bins=15)
            axes[0].set_title("Patient Age Distribution", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Age (Years)")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(axis="y", linestyle="--", alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "Age data unavailable", ha="center", va="center")

        # Panel 2: Anatomical Localization
        if "localization" in self.df.columns:
            loc_counts = self.df["localization"].fillna("unknown").value_counts().head(8)
            sns.barplot(x=loc_counts.values, y=loc_counts.index, ax=axes[1], hue=loc_counts.index, palette="crest", legend=False)
            axes[1].set_title("Anatomical Site Frequency (Top 8)", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Sample Count")
            axes[1].grid(axis="x", linestyle="--", alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Localization data unavailable", ha="center", va="center")

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        plt.close()
        logger.info(f"Demographics chart saved to {out}")

    def generate_markdown_report(self, output_path: Union[str, Path]) -> str:
        """Generate full clinical research dataset audit report in GitHub Flavored Markdown."""
        analysis = self.analyzer.generate_full_analysis_dict()
        class_dist = analysis["class_distribution"]
        mult = analysis["patient_multiplicity"]
        demo = analysis["demographics"]

        lines = [
            "# DermaAI: Clinical Dataset Audit & Distribution Report",
            "",
            "## 1. Executive Summary",
            f"- **Total Dermoscopy Records:** {class_dist['total_samples']:,}",
            f"- **Unique Patient Lesions:** {mult.get('unique_lesions', 'N/A')}",
            f"- **Class Imbalance Ratio (IR):** **{class_dist['imbalance_ratio']} : 1** (Majority: `{class_dist['majority_class']}`, Minority: `{class_dist['minority_class']}`)",
            f"- **Normalized Class Entropy:** {class_dist['normalized_entropy']} / 1.000",
            "",
            "## 2. Diagnosis Class Distribution",
            "",
            "| Diagnosis Code | Clinical Category | Sample Count | Percentage |",
            "| :--- | :--- | :--- | :--- |",
        ]

        class_names = {
            "mel": "Melanoma (Critical)",
            "bcc": "Basal Cell Carcinoma (High)",
            "akiec": "Actinic Keratosis (High)",
            "bkl": "Benign Keratosis (Low)",
            "nv": "Melanocytic Nevus (Low)",
            "df": "Dermatofibroma (Low)",
            "vasc": "Vascular Lesion (Low)",
        }

        for code, count in class_dist["counts"].items():
            pct = class_dist["percentages"].get(code, 0.0)
            name = class_names.get(code, "Unknown")
            lines.append(f"| `{code}` | {name} | {count:,} | {pct:.2f}% |")

        lines.extend([
            "",
            "## 3. Patient Lesion Multiplicity Audit",
            f"- **Single-Image Lesions:** {mult.get('single_image_lesions', 'N/A')} ({round(100 - mult.get('multi_image_percentage', 0), 1)}%)",
            f"- **Multi-Image Lesions:** {mult.get('multi_image_lesions', 'N/A')} ({mult.get('multi_image_percentage', 'N/A')}%)",
            f"- **Max Images for Single Lesion:** {mult.get('max_images_per_lesion', 'N/A')}",
            "> **Medical Integrity Assertion:** Stratified partitioning must strictly group by `lesion_id` to guarantee zero data leakage between train and test splits.",
            "",
            "## 4. Patient Demographics Summary",
        ])

        if "age" in demo:
            age_info = demo["age"]
            lines.extend([
                f"- **Patient Age:** Mean {age_info['mean']} ± {age_info['std']} years (Range: {age_info['min']} - {age_info['max']} yrs, Median: {age_info['median']})",
                f"- **Missing Age Records:** {age_info['missing_samples']}",
            ])

        if "sex" in demo:
            sex_summary = ", ".join([f"{k.capitalize()}: {v}" for k, v in demo["sex"].items()])
            lines.append(f"- **Biological Sex Distribution:** {sex_summary}")

        lines.extend([
            "",
            "---",
            "*Report generated automatically by DermaAI Dataset Reporter.*",
        ])

        report_content = "\n".join(lines)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Markdown report generated at: {out}")
        return report_content
