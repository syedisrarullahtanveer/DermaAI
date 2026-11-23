import React from 'react';

export interface ClassProbability {
  label: string;
  probability: number;
}

interface ClassProbabilityBarChartProps {
  probabilities: ClassProbability[];
}

export const ClassProbabilityBarChart: React.FC<ClassProbabilityBarChartProps> = ({ probabilities }) => {
  return (
    <div className="probability-chart" data-testid="probability-chart">
      {probabilities.map((item) => (
        <div key={item.label} style={{ marginBottom: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span>{item.label}</span>
            <span>{(item.probability * 100).toFixed(1)}%</span>
          </div>
          <div style={{ height: '8px', background: '#e0e0e0', borderRadius: '4px', overflow: 'hidden' }}>
            <div
              style={{
                width: `${Math.min(100, item.probability * 100)}%`,
                height: '100%',
                background: item.probability > 0.5 ? '#d93025' : '#1a73e8',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};
