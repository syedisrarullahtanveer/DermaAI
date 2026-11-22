import React, { useState } from 'react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showGradCam, setShowGradCam] = useState(true);
  const [applyHairRemoval, setApplyHairRemoval] = useState(true);

  // Patient Demographic Form State
  const [patientAge, setPatientAge] = useState('');
  const [patientSex, setPatientSex] = useState('unknown');
  const [anatomicalSite, setAnatomicalSite] = useState('unknown');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    const params = new URLSearchParams({
      apply_hair_removal: applyHairRemoval.toString(),
      include_gradcam: 'true',
    });

    try {
      const response = await fetch(`http://localhost:8000/api/predict?${params.toString()}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Inference API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to DermaAI backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem 1rem', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Header */}
      <header style={{ marginBottom: '2rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: '#0f172a', margin: 0 }}>
              DermaAI <span style={{ fontSize: '1rem', color: '#2563eb', fontWeight: 'normal' }}>v1.0.0</span>
            </h1>
            <p style={{ color: '#64748b', margin: '0.25rem 0 0 0', fontSize: '0.95rem' }}>
              Intelligent Skin Disease Detection with Attention Mechanisms & Explainable AI (Grad-CAM)
            </p>
          </div>
          <div style={{ padding: '0.5rem 1rem', background: '#eff6ff', borderRadius: '0.5rem', color: '#1d4ed8', fontSize: '0.875rem', fontWeight: '500' }}>
            Clinical Research Prototype
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* Left Column: Image Upload & Controls */}
        <section style={{ background: '#ffffff', borderRadius: '0.75rem', border: '1px solid #e2e8f0', padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginTop: 0, marginBottom: '1rem', color: '#1e293b' }}>
            1. Dermoscopic Image Upload
          </h2>

          <div
            style={{
              border: '2px dashed #cbd5e1',
              borderRadius: '0.5rem',
              padding: '2rem',
              textAlign: 'center',
              backgroundColor: '#f8fafc',
              cursor: 'pointer',
              marginBottom: '1.5rem',
            }}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input
              id="file-upload"
              type="file"
              accept="image/jpeg,image/png"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected dermoscopy scan"
                style={{ maxHeight: '240px', maxWidth: '100%', borderRadius: '0.375rem', objectFit: 'contain' }}
              />
            ) : (
              <div>
                <p style={{ margin: 0, fontWeight: '500', color: '#334155' }}>Click or drop dermoscopic lesion image here</p>
                <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.85rem', color: '#94a3b8' }}>Supports JPEG, PNG (HAM10000 / ISIC format)</p>
              </div>
            )}
          </div>

          {/* Clinical Demographics Inputs */}
          <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#334155', marginBottom: '0.75rem' }}>
            Patient Metadata (Optional)
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#64748b', marginBottom: '0.25rem' }}>Age</label>
              <input
                type="number"
                placeholder="Years"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                style={{ width: '100%', padding: '0.4rem 0.6rem', borderRadius: '0.375rem', border: '1px solid #cbd5e1' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#64748b', marginBottom: '0.25rem' }}>Sex</label>
              <select
                value={patientSex}
                onChange={(e) => setPatientSex(e.target.value)}
                style={{ width: '100%', padding: '0.4rem 0.6rem', borderRadius: '0.375rem', border: '1px solid #cbd5e1' }}
              >
                <option value="unknown">Unknown</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#64748b', marginBottom: '0.25rem' }}>Anatomical Site</label>
              <select
                value={anatomicalSite}
                onChange={(e) => setAnatomicalSite(e.target.value)}
                style={{ width: '100%', padding: '0.4rem 0.6rem', borderRadius: '0.375rem', border: '1px solid #cbd5e1' }}
              >
                <option value="unknown">Unknown</option>
                <option value="back">Back</option>
                <option value="lower extremity">Lower Extremity</option>
                <option value="upper extremity">Upper Extremity</option>
                <option value="trunk">Trunk</option>
                <option value="face">Face</option>
                <option value="chest">Chest</option>
              </select>
            </div>
          </div>

          {/* Preprocessing Toggles */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: '#334155' }}>
              <input
                type="checkbox"
                checked={applyHairRemoval}
                onChange={(e) => setApplyHairRemoval(e.target.checked)}
              />
              Enable DullRazor Hair Removal Filter
            </label>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            style={{
              width: '100%',
              padding: '0.75rem 1rem',
              backgroundColor: !selectedFile || loading ? '#94a3b8' : '#2563eb',
              color: '#ffffff',
              border: 'none',
              borderRadius: '0.5rem',
              fontWeight: '600',
              fontSize: '1rem',
              cursor: !selectedFile || loading ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s',
            }}
          >
            {loading ? 'Analyzing Lesion...' : 'Run Diagnostic Analysis'}
          </button>

          {error && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '0.375rem', color: '#b91c1c', fontSize: '0.875rem' }}>
              {error}
            </div>
          )}
        </section>

        {/* Right Column: Diagnostic Results & XAI */}
        <section style={{ background: '#ffffff', borderRadius: '0.75rem', border: '1px solid #e2e8f0', padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginTop: 0, marginBottom: '1rem', color: '#1e293b' }}>
            2. Diagnostic Findings & Explainability
          </h2>

          {!result ? (
            <div style={{ textAlign: 'center', padding: '4rem 1rem', color: '#94a3b8' }}>
              <p>Upload a dermoscopy image and click "Run Diagnostic Analysis" to view predictions and Grad-CAM localization.</p>
            </div>
          ) : (
            <div>
              {/* High Risk Warning Banner */}
              {result.high_risk_alert && (
                <div style={{ padding: '0.75rem 1rem', backgroundColor: '#fff1f2', border: '1px solid #ffe4e6', borderRadius: '0.5rem', color: '#be123c', marginBottom: '1.25rem' }}>
                  <strong>⚠️ Critical Finding:</strong> Malignant/pre-cancerous lesion pattern detected. Immediate clinical histopathology is advised.
                </div>
              )}

              {/* Primary Diagnosis Card */}
              <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', border: '1px solid #e2e8f0', marginBottom: '1.5rem' }}>
                <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 'bold' }}>
                  Primary Predicted Diagnosis
                </span>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.25rem' }}>
                  <h3 style={{ margin: 0, fontSize: '1.25rem', color: '#0f172a' }}>
                    {result.top_prediction.name} ({result.top_prediction.code.toUpperCase()})
                  </h3>
                  <span style={{
                    padding: '0.25rem 0.5rem',
                    borderRadius: '0.25rem',
                    fontSize: '0.8rem',
                    fontWeight: 'bold',
                    backgroundColor: result.top_prediction.risk_level === 'Critical' ? '#fee2e2' : '#f1f5f9',
                    color: result.top_prediction.risk_level === 'Critical' ? '#991b1b' : '#334155',
                  }}>
                    {result.top_prediction.risk_level} Risk
                  </span>
                </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#475569' }}>
                  Model Confidence: <strong>{(result.top_prediction.probability * 100).toFixed(1)}%</strong>
                  <span style={{ marginLeft: '1rem', fontSize: '0.8rem', color: '#94a3b8' }}>
                    (Latency: {result.inference_time_ms} ms)
                  </span>
                </div>
              </div>

              {/* Grad-CAM Visual Explanation */}
              {result.gradcam_base64 && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <h4 style={{ margin: 0, fontSize: '0.95rem', color: '#334155' }}>
                      Explainable AI: Grad-CAM Activation Heatmap
                    </h4>
                    <button
                      onClick={() => setShowGradCam(!showGradCam)}
                      style={{ fontSize: '0.8rem', padding: '0.2rem 0.5rem', border: '1px solid #cbd5e1', borderRadius: '0.25rem', background: '#fff', cursor: 'pointer' }}
                    >
                      {showGradCam ? 'Show Original' : 'Show Thermal Overlay'}
                    </button>
                  </div>
                  <div style={{ textAlign: 'center', backgroundColor: '#000', borderRadius: '0.5rem', padding: '0.5rem' }}>
                    <img
                      src={showGradCam ? `data:image/jpeg;base64,${result.gradcam_base64}` : previewUrl}
                      alt="Grad-CAM Lesion Heatmap"
                      style={{ maxHeight: '200px', maxWidth: '100%', objectFit: 'contain' }}
                    />
                  </div>
                  <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: '#64748b' }}>
                    Warm regions (red/orange) highlight morphological patterns that most strongly drove the model's diagnostic classification.
                  </p>
                </div>
              )}

              {/* Ranked Probabilities List */}
              <div>
                <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.95rem', color: '#334155' }}>
                  Differential Diagnosis Breakdown (7 Classes)
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {result.predictions.map((item) => (
                    <div key={item.code} style={{ fontSize: '0.85rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                        <span style={{ color: '#334155' }}>{item.name}</span>
                        <span style={{ fontWeight: '600' }}>{(item.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                        <div
                          style={{
                            height: '100%',
                            width: `${(item.probability * 100)}%`,
                            backgroundColor: item.risk_level === 'Critical' ? '#ef4444' : '#3b82f6',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
