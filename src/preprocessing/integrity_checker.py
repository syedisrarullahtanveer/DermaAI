"""Dataset Integrity and Corruption Auditing Module for Dermoscopic Imaging."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import setup_logger

logger = setup_logger("DermaAI.IntegrityChecker")


class DatasetIntegrityChecker:
    """Audits image datasets for missing references, corrupted files, and format anomalies."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dirs: List[Union[str, Path]],
        image_id_col: str = "image_id",
    ):
        self.metadata_df = metadata_df.copy()
        self.image_dirs = [Path(d) for d in image_dirs]
        self.image_id_col = image_id_col
        self.indexed_files: Dict[str, Path] = {}
        self._index_filesystem()

    def _index_filesystem(self) -> None:
        """Scan all provided image directories and index image IDs to filesystem paths."""
        self.indexed_files.clear()
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for img_dir in self.image_dirs:
            if not img_dir.exists():
                logger.warning(f"Image directory does not exist: {img_dir}")
                continue

            for file_path in img_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    image_id = file_path.stem
                    self.indexed_files[image_id] = file_path

        logger.info(f"Indexed {len(self.indexed_files)} image files across {len(self.image_dirs)} directories.")

    def check_missing_images(self) -> List[str]:
        """Identify image IDs cataloged in the metadata that do not exist on disk."""
        metadata_ids = self.metadata_df[self.image_id_col].astype(str).tolist()
        missing = [img_id for img_id in metadata_ids if img_id not in self.indexed_files]
        return missing

    def check_orphan_images(self) -> List[str]:
        """Identify image files present on disk that are not registered in the metadata."""
        metadata_ids_set = set(self.metadata_df[self.image_id_col].astype(str))
        orphans = [img_id for img_id in self.indexed_files.keys() if img_id not in metadata_ids_set]
        return orphans

    def audit_image_file(self, file_path: Path) -> Dict[str, Union[bool, str, Tuple]]:
        """Perform deep integrity inspection on a single image file."""
        res = {
            "file_path": str(file_path),
            "is_valid": True,
            "error": None,
            "channels": 0,
            "dimensions": (0, 0),
            "file_size_bytes": 0,
        }

        # 1. Zero-byte check
        try:
            size = file_path.stat().st_size
            res["file_size_bytes"] = size
            if size == 0:
                res["is_valid"] = False
                res["error"] = "ZeroByteFile"
                return res
        except Exception as e:
            res["is_valid"] = False
            res["error"] = f"StatError: {str(e)}"
            return res

        # 2. PIL format & header verification
        try:
            with Image.open(file_path) as pil_img:
                pil_img.verify()
        except Exception as e:
            res["is_valid"] = False
            res["error"] = f"CorruptedHeader: {str(e)}"
            return res

        # 3. OpenCV decompression & channel validation
        try:
            img_bgr = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                res["is_valid"] = False
                res["error"] = "DecodeFailure"
                return res

            if img_bgr.ndim == 2:
                res["channels"] = 1
                res["dimensions"] = img_bgr.shape
            elif img_bgr.ndim == 3:
                res["channels"] = img_bgr.shape[2]
                res["dimensions"] = (img_bgr.shape[0], img_bgr.shape[1])
            else:
                res["is_valid"] = False
                res["error"] = f"AbnormalTensorRank: {img_bgr.ndim}"
                return res

            # Ensure valid 3-channel RGB image
            if res["channels"] != 3:
                res["is_valid"] = False
                res["error"] = f"InvalidChannels: Expected 3 channels, found {res['channels']}"

        except Exception as e:
            res["is_valid"] = False
            res["error"] = f"OpenCVDecodeError: {str(e)}"

        return res

    def run_full_audit(self, max_samples: Optional[int] = None) -> Dict:
        """Execute complete integrity audit across all registered images."""
        logger.info("Starting dataset integrity audit...")
        missing_images = self.check_missing_images()
        orphan_images = self.check_orphan_images()

        corrupted_files = []
        valid_count = 0
        dimension_summary = {}

        images_to_check = list(self.indexed_files.items())
        if max_samples and max_samples < len(images_to_check):
            images_to_check = images_to_check[:max_samples]

        for img_id, file_path in images_to_check:
            audit = self.audit_image_file(file_path)
            if not audit["is_valid"]:
                corrupted_files.append({"image_id": img_id, **audit})
            else:
                valid_count += 1
                dims = audit["dimensions"]
                dimension_summary[dims] = dimension_summary.get(dims, 0) + 1

        audit_report = {
            "total_metadata_records": len(self.metadata_df),
            "total_filesystem_images": len(self.indexed_files),
            "missing_from_disk_count": len(missing_images),
            "missing_image_ids": missing_images[:50], # Sample of up to 50
            "orphan_images_count": len(orphan_images),
            "audited_files_count": len(images_to_check),
            "valid_files_count": valid_count,
            "corrupted_files_count": len(corrupted_files),
            "corrupted_files": corrupted_files,
            "dimension_distributions": {f"{k[0]}x{k[1]}": v for k, v in dimension_summary.items()},
        }

        logger.info(
            f"Audit finished: {valid_count} valid, {len(corrupted_files)} corrupted, "
            f"{len(missing_images)} missing, {len(orphan_images)} orphans."
        )
        return audit_report
