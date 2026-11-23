import React from 'react';

interface UncertaintyGaugeProps {
  uncertaintyScore: number; // [0, 1]
  conformalClasses: string[];
}

export const UncertaintyGauge: React.FC<UncertaintyGaugeProps> = ({ uncertaintyScore, conformalClasses }) => {
  const isHighUncertainty = uncertaintyScore > 0.35;

  return (
    <div className="uncertainty-gauge" data-testid="uncertainty-gauge" style={{ padding: '8px', border: '1px solid #eee' }}>
      <h4>Prediction Confidence Analysis</h4>
      <p>Epistemic Uncertainty: {(uncertaintyScore * 100).toFixed(1)}%</p>
      {isHighUncertainty && (
        <div style={{ color: '#b06000', fontWeight: 'bold' }}>
          ⚠️ High Diagnostic Ambiguity: Consider expert second opinion or histology.
        </div>
      )}
      <p>90% Conformal Coverage Set: [{conformalClasses.join(', ')}]</p>
    </div>
  );
};
