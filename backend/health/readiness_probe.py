"""
Model readiness probe reporting loaded status and device health.
"""
class ReadinessProbe:
    def __init__(self):
        self.model_loaded = False
        self.weights_hash = None

    def mark_ready(self, weights_hash: str):
        self.model_loaded = True
        self.weights_hash = weights_hash

    def check_health(self) -> dict:
        return {
            "status": "ready" if self.model_loaded else "warming_up",
            "model_loaded": self.model_loaded,
            "weights_hash": self.weights_hash
        }
