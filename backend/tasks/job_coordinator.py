"""
Async prediction job lifecycle manager and status coordinator.
"""
import time

class JobCoordinator:
    def __init__(self):
        self.jobs = {}

    def create_job(self, job_id: str) -> dict:
        job = {"status": "pending", "progress": 0, "created_at": time.time(), "result": None}
        self.jobs[job_id] = job
        return job

    def update_progress(self, job_id: str, progress: int, status: str = "processing"):
        if job_id in self.jobs:
            self.jobs[job_id]["progress"] = progress
            self.jobs[job_id]["status"] = status

    def complete_job(self, job_id: str, result: dict):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["result"] = result

    def get_job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)
