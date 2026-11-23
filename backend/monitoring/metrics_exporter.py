"""
Prometheus formatted metrics exporter for inference latencies and triage counts.
"""
class MetricsExporter:
    def __init__(self):
        self.prediction_counts = {}
        self.latencies = []

    def record_inference(self, predicted_class: str, latency_ms: float):
        self.prediction_counts[predicted_class] = self.prediction_counts.get(predicted_class, 0) + 1
        self.latencies.append(latency_ms)

    def generate_prometheus_metrics(self) -> str:
        lines = [
            "# HELP dermaai_inferences_total Total lesion classification queries.",
            "# TYPE dermaai_inferences_total counter"
        ]
        for cls, count in self.prediction_counts.items():
            lines.append(f'dermaai_inferences_total{{class="{cls}"}} {count}')
            
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        lines.extend([
            "# HELP dermaai_latency_ms Mean inference latency in milliseconds.",
            "# TYPE dermaai_latency_ms gauge",
            f"dermaai_latency_ms {avg_latency:.2f}"
        ])
        return "\n".join(lines) + "\n"
