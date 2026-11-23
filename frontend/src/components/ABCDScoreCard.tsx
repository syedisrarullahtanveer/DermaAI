import React from 'react';

interface ABCDScoreCardProps {
  asymmetryScore: number;
  borderIrregularity: number;
  colorVariegation: number;
  diameterMm: number;
}

export const ABCDScoreCard: React.FC<ABCDScoreCardProps> = ({
  asymmetryScore,
  borderIrregularity,
  colorVariegation,
  diameterMm,
}) => {
  // Total Dermatoscopy Value (TDV) formula = A*1.3 + B*0.1 + C*0.5 + D*0.5
  const tdv = asymmetryScore * 1.3 + borderIrregularity * 0.1 + colorVariegation * 0.5 + diameterMm * 0.5;

  return (
    <div className="abcd-score-card" data-testid="abcd-card" style={{ padding: '12px', border: '1px solid #ddd' }}>
      <h3>ABCD Clinical Scorecard</h3>
      <ul>
        <li><strong>A - Asymmetry:</strong> {asymmetryScore.toFixed(1)}</li>
        <li><strong>B - Border:</strong> {borderIrregularity.toFixed(1)}</li>
        <li><strong>C - Color:</strong> {colorVariegation} / 6</li>
        <li><strong>D - Diameter:</strong> {diameterMm.toFixed(1)} mm {diameterMm >= 6.0 ? '(⚠️ > 6mm)' : ''}</li>
      </ul>
      <p><strong>Total Dermatoscopy Value (TDV):</strong> {tdv.toFixed(2)}</p>
    </div>
  );
};
