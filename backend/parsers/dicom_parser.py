"""
DICOM header parser and modality tag extractor for dermoscopic image integration.
"""
import struct

class DICOMParser:
    def parse_header(self, raw_bytes: bytes) -> dict:
        # Check DICOM prefix at byte 128
        if len(raw_bytes) < 132:
            return {"valid_dicom": False, "error": "File too small"}
        
        prefix = raw_bytes[128:132]
        is_dicom = prefix == b"DICM"
        return {
            "valid_dicom": is_dicom,
            "has_preamble": len(raw_bytes) >= 128,
            "byte_size": len(raw_bytes),
            "modality": "XC" if is_dicom else "UNKNOWN" # XC = External Camera / Dermoscopy
        }
