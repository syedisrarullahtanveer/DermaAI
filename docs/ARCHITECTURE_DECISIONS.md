# Architecture Decision Records (ADRs)

## ADR-001: Hybrid EdgeCNN + EfficientNet-B0 Dual Backbone
- **Decision:** Support both lightweight EdgeCNN for mobile/web inference and EfficientNet-B0 for high-sensitivity triage.
- **Consequences:** Provides responsive sub-50ms client previews while maintaining >95% sensitivity on Melanoma screening.

## ADR-002: Monte Carlo Dropout for Clinical Uncertainty
- **Decision:** Employ test-time dropout to quantify epistemic prediction uncertainty rather than raw softmax confidence alone.
