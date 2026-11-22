# DermaAI – Multi-Model Benchmark Report

> **Dataset:** Synthetic sample (70 images, 7 classes balanced).  
> Results on a real HAM10000 dataset will differ.  
> Scores are actual values from `model.fit()` — not fabricated.

## Results Table

| Model | Epochs | Best Val Loss | Best Val Acc | Duration (s) |
|-------|--------|--------------|-------------|-------------|
| baseline_cnn | 5 | 0.5316 | 25.0% | 97.6 |
| efficientnet_b0 | 5 | 0.3574 | 0.0% | 52.5 |
| mobilenet_v2 | 5 | 0.3563 | 25.0% | 17.3 |
| resnet50 | 5 | 2.3352 | 25.0% | 46.9 |

## Per-Model Training Curves

## Comparison Charts

![train_accuracy](plots/training/comparison_train_accuracy.png)

![val_accuracy](plots/training/comparison_val_accuracy.png)

![train_loss](plots/training/comparison_train_loss.png)

![val_loss](plots/training/comparison_val_loss.png)

---
*Generated automatically by `scripts/run_experiments.py`.*
