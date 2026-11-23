"""
Pydantic schemas for structured clinical dermoscopy query metadata.
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ClinicalQueryMetadata:
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = "unknown"
    anatomical_site: Optional[str] = "unknown"
    lesion_evolution: Optional[str] = "stable" # stable, enlarging, bleeding, itching
    previous_melanoma_history: bool = False

    def validate(self) -> list:
        errors = []
        if self.patient_age is not None and (self.patient_age < 0 or self.patient_age > 120):
            errors.append("patient_age must be between 0 and 120")
        if self.patient_sex and self.patient_sex.lower() not in ["male", "female", "unknown"]:
            errors.append("patient_sex must be male, female, or unknown")
        return errors
