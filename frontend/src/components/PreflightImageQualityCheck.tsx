import React from 'react';

interface PreflightCheckProps {
  resolutionOk: boolean;
  blurScore: number;
  minBlurThreshold?: number;
}

export const PreflightImageQualityCheck: React.FC<PreflightCheckProps> = ({
  resolutionOk,
  blurScore,
  minBlurThreshold = 100,
}) => {
  const isBlurry = blurScore < minBlurThreshold;

  return (
    <div className="preflight-status" data-testid="preflight-check" style={{ padding: '12px', border: '1px solid #ccc' }}>
      <h4>Pre-flight Image Quality</h4>
      <p>Resolution: {resolutionOk ? '✅ Pass (224x224+)' : '⚠️ Warning: Low resolution'}</p>
      <p>Sharpness: {!isBlurry ? `✅ Sharp (${blurScore.toFixed(0)})` : `⚠️ Blurry (${blurScore.toFixed(0)} < ${minBlurThreshold})`}</p>
    </div>
  );
};
