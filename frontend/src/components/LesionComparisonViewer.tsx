import React, { useState } from 'react';

interface LesionComparisonViewerProps {
  originalImage: string;
  gradcamOverlay: string;
  lesionLabel?: string;
  confidence?: number;
}

export const LesionComparisonViewer: React.FC<LesionComparisonViewerProps> = ({
  originalImage,
  gradcamOverlay,
  lesionLabel = 'Melanocytic Lesion',
  confidence = 0.85,
}) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side'>('slider');

  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-slate-100 max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-800">{lesionLabel}</h3>
          <p className="text-sm text-slate-500">
            Confidence: <span className="font-semibold text-indigo-600">{(confidence * 100).toFixed(1)}%</span>
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('slider')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition ${
              viewMode === 'slider' ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Split Slider
          </button>
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition ${
              viewMode === 'side-by-side' ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Side by Side
          </button>
        </div>
      </div>

      {viewMode === 'slider' ? (
        <div className="relative w-full aspect-square overflow-hidden rounded-lg border border-slate-200 select-none">
          {/* Base Layer: Grad-CAM Saliency */}
          <img
            src={gradcamOverlay}
            alt="Grad-CAM Overlay"
            className="absolute inset-0 w-full h-full object-cover"
          />
          {/* Top Layer: Original Dermoscopic Image clipped by slider */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ width: `${sliderPosition}%` }}
          >
            <img
              src={originalImage}
              alt="Original Dermoscopy"
              className="absolute inset-0 w-full h-full object-cover max-w-none"
              style={{ width: '100%', height: '100%' }}
            />
          </div>
          {/* Divider line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
            style={{ left: `${sliderPosition}%` }}
          />
          {/* Range Slider Control */}
          <input
            type="range"
            min="0"
            max="100"
            value={sliderPosition}
            onChange={(e) => setSliderPosition(Number(e.target.value))}
            className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize"
            aria-label="Comparison split slider"
          />
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg overflow-hidden border border-slate-200">
            <span className="block text-xs font-medium text-slate-500 bg-slate-50 px-2 py-1 border-b">
              Original Dermoscopy
            </span>
            <img src={originalImage} alt="Original" className="w-full aspect-square object-cover" />
          </div>
          <div className="rounded-lg overflow-hidden border border-slate-200">
            <span className="block text-xs font-medium text-slate-500 bg-slate-50 px-2 py-1 border-b">
              Grad-CAM Explanation
            </span>
            <img src={gradcamOverlay} alt="Grad-CAM" className="w-full aspect-square object-cover" />
          </div>
        </div>
      )}

      <p className="mt-3 text-xs text-center text-slate-400">
        Slide horizontally to compare raw dermoscopic pigment patterns with deep saliency localization.
      </p>
    </div>
  );
};
