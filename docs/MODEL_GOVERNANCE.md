# Model Governance & Continuous Monitoring

## Performance Drift Detection
- Weekly population stability index (PSI) computation on prediction distributions.
- Automatic alerting if mean predicted melanoma probability deviates > 15% from baseline prevalence.

## Retraining Safeguards
- Automated shadow-model deployment with side-by-side evaluation before traffic cutover.
