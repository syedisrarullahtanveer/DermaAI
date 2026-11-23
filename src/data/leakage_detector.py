"""
Patient-level data leakage detector ensuring lesions from the same patient do not appear in both train and validation sets.
"""
from typing import Set

class PatientLeakageDetector:
    def detect_leakage(self, train_patient_ids: list, val_patient_ids: list) -> Set[str]:
        set_train = set(train_patient_ids)
        set_val = set(val_patient_ids)
        return set_train.intersection(set_val)

    def is_clean(self, train_patient_ids: list, val_patient_ids: list) -> bool:
        return len(self.detect_leakage(train_patient_ids, val_patient_ids)) == 0
