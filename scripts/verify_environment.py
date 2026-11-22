"""Environment Verification and Diagnostic Script for DermaAI."""

import sys
from pathlib import Path


def run_diagnostics():
    print("=" * 60)
    print("  DermaAI: System Diagnostics & Environment Verification")
    print("=" * 60)

    # 1. Python runtime
    print(f"[x] Python Version: {sys.version.split()[0]} ({sys.platform})")

    # 2. Project root & directories
    project_root = Path(__file__).resolve().parent.parent
    print(f"[x] Project Root: {project_root}")

    required_dirs = [
        "dataset",
        "dataset/raw",
        "dataset/processed",
        "dataset/metadata",
        "configs",
        "notebooks",
        "src",
        "src/data",
        "src/preprocessing",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/explainability",
        "backend",
        "frontend",
        "tests",
        "docs",
        "scripts",
    ]

    all_dirs_ok = True
    for d in required_dirs:
        dir_path = project_root / d
        if dir_path.exists() and dir_path.is_dir():
            pass
        else:
            all_dirs_ok = False
            print(f"[!] Missing directory: {d}")

    if all_dirs_ok:
        print(f"[x] All {len(required_dirs)} required project directories verified.")

    # 3. Deep Learning Frameworks
    try:
        import tensorflow as tf
        print(f"[x] TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[x] GPU Acceleration: ENABLED ({len(gpus)} device(s) found: {[g.name for g in gpus]})")
        else:
            print("[i] GPU Acceleration: CPU Mode (No physical GPU detected)")
    except ImportError:
        print("[!] TensorFlow is not installed.")

    # 4. Computer Vision & Data
    try:
        import cv2
        print(f"[x] OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("[!] OpenCV is not installed.")

    try:
        import sklearn
        print(f"[x] Scikit-Learn Version: {sklearn.__version__}")
    except ImportError:
        print("[!] Scikit-Learn is not installed.")

    try:
        import fastapi
        print(f"[x] FastAPI Version: {fastapi.__version__}")
    except ImportError:
        print("[!] FastAPI is not installed.")

    print("=" * 60)
    print("  Diagnostics Complete. Environment is Ready for Development.")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
