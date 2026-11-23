"""
Audit logging for clinical prediction compliance and traceability.
"""
import json
import os
import time

class AuditLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log_prediction_event(self, request_id: str, client_ip: str, model_version: str, predicted_class: str, confidence: float):
        record = {
            "timestamp": time.time(),
            "request_id": request_id,
            "client_ip": client_ip,
            "model_version": model_version,
            "predicted_class": predicted_class,
            "confidence": confidence
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
