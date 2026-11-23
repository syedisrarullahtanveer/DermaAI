"""
Per-epoch metric snapshot logger and CSV/JSON exporter.
"""
import csv
import json
import os

class MetricSnapshotLogger:
    def __init__(self):
        self.history = []

    def record_epoch(self, epoch: int, metrics: dict):
        record = {"epoch": epoch}
        record.update(metrics)
        self.history.append(record)

    def export_csv(self, filepath: str):
        if not self.history:
            return
        keys = list(self.history[0].keys())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def export_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
