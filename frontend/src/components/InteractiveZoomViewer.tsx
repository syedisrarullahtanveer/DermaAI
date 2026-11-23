import React, { useState } from 'react';

interface InteractiveZoomViewerProps {
  imageSrc: string;
  heatmapSrc?: string;
}

export const InteractiveZoomViewer: React.FC<InteractiveZoomViewerProps> = ({ imageSrc, heatmapSrc }) => {
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);

  return (
    <div className="interactive-zoom-viewer" data-testid="zoom-viewer" style={{ position: 'relative', overflow: 'hidden' }}>
      <div style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left', transition: 'transform 0.2s ease' }}>
        <img src={imageSrc} alt="Dermoscopy Lesion" style={{ width: '100%', display: 'block' }} />
        {heatmapSrc && showHeatmap && (
          <img
            src={heatmapSrc}
            alt="Grad-CAM Overlay"
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', opacity: 0.5, pointerEvents: 'none' }}
          />
        )}
      </div>
      <div className="zoom-controls" style={{ marginTop: '8px' }}>
        <button onClick={() => setZoomLevel((z) => Math.min(3, z + 0.5))}>Zoom In</button>
        <button onClick={() => setZoomLevel((z) => Math.max(1, z - 0.5))}>Zoom Out</button>
        {heatmapSrc && (
          <button onClick={() => setShowHeatmap((s) => !s)}>{showHeatmap ? 'Hide Overlay' : 'Show Overlay'}</button>
        )}
      </div>
    </div>
  );
};
