"""
Server-Sent Events (SSE) progress frame formatter for deep inference pipelines.
"""
import json

class SSEProgressGenerator:
    def format_event(self, step_name: str, progress_percent: int, message: str) -> str:
        data = {
            "step": step_name,
            "progress": min(100, max(0, progress_percent)),
            "message": message
        }
        return f"event: progress\ndata: {json.dumps(data)}\n\n"

    def format_complete(self, result_dict: dict) -> str:
        return f"event: complete\ndata: {json.dumps(result_dict)}\n\n"
