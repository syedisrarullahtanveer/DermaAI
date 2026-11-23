import React from 'react';

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

interface ClinicalRiskBadgeProps {
  level: RiskLevel;
  score?: number;
}

export const ClinicalRiskBadge: React.FC<ClinicalRiskBadgeProps> = ({ level, score }) => {
  const styles: Record<RiskLevel, { bg: string; text: string; label: string }> = {
    low: { bg: '#e6f4ea', text: '#137333', label: 'Low Risk' },
    moderate: { bg: '#fef7e0', text: '#b06000', label: 'Moderate Risk' },
    high: { bg: '#feefe3', text: '#c5221f', label: 'High Risk' },
    critical: { bg: '#fce8e6', text: '#a50e0e', label: 'Urgent Biopsy Required' },
  };

  const current = styles[level] || styles.low;

  return (
    <span
      data-testid="clinical-risk-badge"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '4px 12px',
        borderRadius: '16px',
        backgroundColor: current.bg,
        color: current.text,
        fontWeight: 600,
        fontSize: '0.875rem',
      }}
    >
      {current.label} {score !== undefined ? `(${Math.round(score * 100)}%)` : ''}
    </span>
  );
};
