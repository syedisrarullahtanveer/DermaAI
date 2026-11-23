"""
Diagnostic Odds Ratio (DOR) and Positive/Negative Likelihood Ratios (LR+, LR-) for evidence-based dermatology.
"""
class DiagnosticRatioCalculator:
    def compute(self, tp: int, fp: int, fn: int, tn: int, eps: float = 1e-4) -> dict:
        sens = (tp + eps) / (tp + fn + eps)
        spec = (tn + eps) / (tn + fp + eps)
        
        lr_plus = sens / (1.0 - spec + eps)
        lr_minus = (1.0 - sens) / (spec + eps)
        dor = lr_plus / (lr_minus + eps)
        
        return {
            "sensitivity": float(sens),
            "specificity": float(spec),
            "lr_positive": float(lr_plus),
            "lr_negative": float(lr_minus),
            "diagnostic_odds_ratio": float(dor)
        }
