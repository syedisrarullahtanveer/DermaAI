"""Synthetic Clinical Sample Dataset Generator for Testing & Verification.

Generates a lightweight, structurally compliant set of synthetic dermoscopy images
and clinical metadata to enable full-cycle testing without downloading the full 3GB HAM10000 archive.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import cv2
import numpy as np
import pandas as pd

from src.data.splitter import PatientAwareDataSplitter
from src.evaluation.dataset_reporter import DatasetReporter
from src.preprocessing.integrity_checker import DatasetIntegrityChecker
from src.utils import setup_logger

logger = setup_logger("DermaAI.SampleGenerator")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_TYPES = ["histo", "follow_up", "consensus", "confocal"]
SEXES = ["male", "female"]
LOCALIZATIONS = ["back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest"]


def generate_synthetic_lesion_image(
    dx: str,
    width: int = 224,
    height: int = 224,
    add_hair: bool = True,
) -> np.ndarray:
    """Create a synthetic dermoscopy image with realistic skin background and pigmented lesion morphology."""
    # 1. Base skin tone (peach/beige with subtle gaussian texture)
    base_color = np.array([random.randint(180, 220), random.randint(140, 180), random.randint(120, 160)], dtype=np.uint8)
    image = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add subtle skin texture noise
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Lesion parameters (center, size, rotation, asymmetry based on diagnosis)
    cx = width // 2 + random.randint(-15, 15)
    cy = height // 2 + random.randint(-15, 15)
    axis_x = random.randint(30, 60)
    axis_y = random.randint(25, 55)
    angle = random.randint(0, 180)

    # Malignant melanoma ('mel') has darker, irregular, multi-tone pigmentation
    if dx == "mel":
        lesion_color = (random.randint(30, 70), random.randint(20, 50), random.randint(20, 50))
    elif dx == "vasc":
        # Vascular lesions have reddish/purple hue
        lesion_color = (random.randint(120, 170), random.randint(20, 50), random.randint(40, 80))
    elif dx in ["bcc", "akiec"]:
        # Carcinoma / keratosis: pinkish to light brown
        lesion_color = (random.randint(90, 140), random.randint(60, 100), random.randint(50, 80))
    else:
        # Benign nevi ('nv', 'bkl', 'df'): uniform brown
        lesion_color = (random.randint(60, 110), random.randint(40, 80), random.randint(30, 60))

    # Draw elliptical pigmented lesion core
    cv2.ellipse(image, (cx, cy), (axis_x, axis_y), angle, 0, 360, lesion_color, -1)

    # Blur lesion boundary for organic transition
    image = cv2.GaussianBlur(image, (15, 15), 0)

    # 3. Add simulated dark occluding hairs
    if add_hair:
        for _ in range(random.randint(3, 8)):
            pt1 = (random.randint(0, width), random.randint(0, height))
            pt2 = (random.randint(0, width), random.randint(0, height))
            hair_color = (random.randint(20, 40), random.randint(20, 40), random.randint(20, 40))
            cv2.line(image, pt1, pt2, hair_color, thickness=random.choice([1, 2]))

    # 4. Optional subtle vignette / circular dark border simulating dermatoscope lens barrel
    vignette = np.zeros((height, width), dtype=np.float32)
    cv2.circle(vignette, (width // 2, height // 2), int(min(width, height) * 0.65), 1.0, -1)
    vignette = cv2.GaussianBlur(vignette, (45, 45), 0)
    image = (image * vignette[:, :, np.newaxis]).astype(np.uint8)

    return image


def create_sample_dataset(
    output_dir: Path,
    samples_per_class: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate complete sample dataset with images, metadata CSV, and verified splits."""
    random.seed(seed)
    np.random.seed(seed)

    raw_img_dir = output_dir / "raw" / "sample_images"
    metadata_dir = output_dir / "metadata"
    splits_dir = output_dir / "splits"

    raw_img_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    total_samples = len(CLASSES) * samples_per_class
    logger.info(f"Generating {total_samples} synthetic dermoscopy samples ({samples_per_class} per class)...")

    counter = 1
    lesion_counter = 1

    for dx in CLASSES:
        for i in range(samples_per_class):
            img_id = f"ISIC_SAMPLE_{counter:05d}"
            # Introduce multi-image lesions for realistic patient multiplicity
            if i % 3 == 0:
                lesion_id = f"HAM_SAMPLE_{lesion_counter:04d}"
                lesion_counter += 1
            else:
                lesion_id = f"HAM_SAMPLE_{lesion_counter - 1:04d}"

            img_array = generate_synthetic_lesion_image(dx=dx)
            file_path = raw_img_dir / f"{img_id}.jpg"
            cv2.imwrite(str(file_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            record = {
                "lesion_id": lesion_id,
                "image_id": img_id,
                "dx": dx,
                "dx_type": random.choice(DX_TYPES),
                "age": float(random.choice([25, 30, 45, 52, 60, 68, 75, np.nan])),
                "sex": random.choice(SEXES),
                "localization": random.choice(LOCALIZATIONS),
            }
            metadata_records.append(record)
            counter += 1

    df = pd.DataFrame(metadata_records)
    meta_csv_path = metadata_dir / "HAM10000_metadata.csv"
    df.to_csv(meta_csv_path, index=False)
    logger.info(f"Sample metadata saved to: {meta_csv_path}")

    # Run integrity check
    checker = DatasetIntegrityChecker(df, [raw_img_dir])
    audit = checker.run_full_audit()
    logger.info(f"Audit Status: {audit['valid_files_count']}/{audit['total_filesystem_images']} files verified valid.")

    # Partition into patient-aware splits
    splitter = PatientAwareDataSplitter(df, seed=seed)
    train_df, val_df, test_df = splitter.split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splitter.export_splits(train_df, val_df, test_df, output_dir=splits_dir)

    # Generate reporting charts
    reports_dir = output_dir.parent / "reports"
    reporter = DatasetReporter(df)
    reporter.plot_class_distribution(reports_dir / "sample_class_distribution.png")
    reporter.plot_demographics(reports_dir / "sample_demographics_distribution.png")
    reporter.generate_markdown_report(reports_dir / "sample_dataset_report.md")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    ds_dir = project_root / "dataset"
    create_sample_dataset(ds_dir, samples_per_class=10)
