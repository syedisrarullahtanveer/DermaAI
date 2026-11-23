import React from 'react';

export interface ThumbnailItem {
  id: string;
  date: string;
  thumbnailUrl: string;
  riskLevel: string;
}

interface MultiViewGridProps {
  items: ThumbnailItem[];
  onSelect: (id: string) => void;
}

export const MultiViewThumbnailGrid: React.FC<MultiViewGridProps> = ({ items, onSelect }) => {
  return (
    <div className="thumbnail-grid" data-testid="multiview-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: '8px' }}>
      {items.map((item) => (
        <div key={item.id} onClick={() => onSelect(item.id)} style={{ cursor: 'pointer', textAlign: 'center' }}>
          <img src={item.thumbnailUrl} alt={item.date} style={{ width: '100%', borderRadius: '4px' }} />
          <span style={{ fontSize: '0.75rem' }}>{item.date}</span>
        </div>
      ))}
    </div>
  );
};
