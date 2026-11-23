import React from 'react';

interface ClinicalReportSummaryProps {
  caseId: string;
  primaryDiagnosis: string;
  confidence: number;
  recommendation: string;
}

export const ClinicalReportSummary: React.FC<ClinicalReportSummaryProps> = ({
  caseId,
  primaryDiagnosis,
  confidence,
  recommendation,
}) => {
  return (
    <section className="report-summary" data-testid="report-summary" style={{ padding: '16px', background: '#fafafa' }}>
      <h3>Clinical Diagnostic Summary</h3>
      <p><strong>Case ID:</strong> {caseId}</p>
      <p><strong>Primary AI Finding:</strong> {primaryDiagnosis} ({(confidence * 100).toFixed(1)}%)</p>
      <p><strong>Recommended Action:</strong> {recommendation}</p>
      <small style={{ color: '#777' }}>* DermaAI is a clinical decision support tool and does not replace histopathological biopsy.</small>
    </section>
  );
};
