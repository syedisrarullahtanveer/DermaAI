"""Unit Tests for Dataset Reporter."""

from pathlib import Path
import pandas as pd
from src.evaluation.dataset_reporter import DatasetReporter


def test_dataset_reporter(tmp_path):
    """Verify class distribution plot, demographic plot, and markdown report generation."""
    sample_df = pd.DataFrame({
        "image_id": [f"IMG_{i:03d}" for i in range(12)],
        "lesion_id": [f"L_{i // 2:02d}" for i in range(12)],
        "dx": ["nv"] * 6 + ["mel"] * 3 + ["bcc"] * 2 + ["df"] * 1,
        "age": [35.0, 42.0, 50.0, 68.0, 22.0, 55.0, 71.0, 49.0, 60.0, 39.0, 58.0, 63.0],
        "sex": ["male", "female"] * 6,
        "localization": ["back", "face", "trunk"] * 4,
    })

    reporter = DatasetReporter(sample_df)

    # 1. Test class plot
    class_plot = tmp_path / "class_dist.png"
    reporter.plot_class_distribution(class_plot)
    assert class_plot.exists()

    # 2. Test demographic plot
    demo_plot = tmp_path / "demo_dist.png"
    reporter.plot_demographics(demo_plot)
    assert demo_plot.exists()

    # 3. Test markdown report
    md_report = tmp_path / "report.md"
    content = reporter.generate_markdown_report(md_report)
    assert md_report.exists()
    assert "DermaAI: Clinical Dataset Audit" in content
    assert "Melanoma (Critical)" in content
