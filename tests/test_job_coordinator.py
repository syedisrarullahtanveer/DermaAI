import unittest
from backend.tasks.job_coordinator import JobCoordinator

class TestJobCoordinator(unittest.TestCase):
    def test_job_flow(self):
        coord = JobCoordinator()
        coord.create_job("job_01")
        coord.update_progress("job_01", 50)
        self.assertEqual(coord.get_job("job_01")["progress"], 50)
        coord.complete_job("job_01", {"melanoma_risk": 0.89})
        self.assertEqual(coord.get_job("job_01")["status"], "completed")

if __name__ == '__main__':
    unittest.main()
