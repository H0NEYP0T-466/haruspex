import { useState, useCallback, useRef } from 'react';
import api, { type PredictionReport, type EvidenceItem, type ShapFeature } from '../services/api';

function getSeverityClass(severity: number): string {
  if (severity >= 0.8) return 'severity-critical';
  if (severity >= 0.6) return 'severity-high';
  if (severity >= 0.4) return 'severity-medium';
  if (severity >= 0.2) return 'severity-low';
  return 'severity-info';
}

function getSeverityLabel(severity: number): string {
  if (severity >= 0.8) return 'CRITICAL';
  if (severity >= 0.6) return 'HIGH';
  if (severity >= 0.4) return 'MEDIUM';
  if (severity >= 0.2) return 'LOW';
  return 'INFO';
}

function getSourceBadge(source: string) {
  const map: Record<string, { label: string; cls: string }> = {
    correlated: { label: 'Correlated', cls: 'badge-correlated' },
    ai_explanation: { label: 'AI', cls: 'badge-ai' },
    forensic_analysis: { label: 'Forensic', cls: 'badge-forensic' },
  };
  const info = map[source] || { label: source, cls: '' };
  return <span className={`evidence-source-badge ${info.cls}`}>{info.label}</span>;
}

interface UploadPageProps {
  modelGroup: string;
}

const MODEL_ALGORITHMS = [
  { id: 'xgboost', label: 'XGBoost', desc: 'Gradient Boosted Trees (ML Baseline)' },
  { id: 'lightgbm', label: 'LightGBM', desc: 'Fast Gradient Boosting (ML Baseline)' },
  { id: 'mlp', label: 'Group-MLP', desc: 'Group-wise Residual MLP (Deep Learning)' },
  { id: 'tabnet', label: 'TabNet', desc: 'Sequential Attention Network (Deep Learning)' },
];

export default function UploadPage({ modelGroup }: UploadPageProps) {
  const [report, setReport] = useState<PredictionReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('xgboost');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const groupLabel = modelGroup === 'ember' ? 'EMBER' : 'BODMAS';

  const handleAnalyze = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const result = await api.predictFromFile(file, selectedModel, modelGroup);
      setReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [modelGroup, selectedModel]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) handleAnalyze(file);
  }, [handleAnalyze]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleAnalyze(file);
  }, [handleAnalyze]);

  // Analyze a random dataset sample
  const handleRandomAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await api.explainRandom(selectedModel, modelGroup);
      setReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Random analysis failed. Is the API running?');
    } finally {
      setLoading(false);
    }
  }, [modelGroup, selectedModel]);

  return (
    <div>
      <div className="page-header">
        <h2>Malware Analysis</h2>
        <p>Upload a PE file or analyze a random dataset sample</p>
        <div className="active-model-badge">
          <span>Using <strong>{groupLabel} Dataset ({selectedModel.toUpperCase()})</strong></span>
        </div>
      </div>

      {/* Model Selector Bar */}
      {!report && !loading && (
        <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
          <div className="card-header" style={{ marginBottom: 'var(--space-sm)' }}>
            <div className="card-title">Select Model Architecture</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-sm)' }}>
            {MODEL_ALGORITHMS.map((algo) => (
              <button
                key={algo.id}
                type="button"
                className={`model-group-option ${selectedModel === algo.id ? 'active' : ''}`}
                onClick={() => setSelectedModel(algo.id)}
                style={{
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-color)',
                  padding: 'var(--space-sm) var(--space-md)',
                }}
              >
                <div className="model-group-info">
                  <span className="model-group-name">{algo.label}</span>
                  <span className="model-group-desc">{algo.desc}</span>
                </div>
                {selectedModel === algo.id && <span className="model-group-check">✓</span>}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Upload zone */}
      {!report && !loading && (
        <>
          <div
            className={`dropzone ${dragActive ? 'active' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">Drop file</div>
            <h3>Drop a PE file here or click to browse</h3>
            <p>Supports Windows PE executables (.exe, .dll)</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".exe,.dll,.sys,.pe"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>

          <div style={{ textAlign: 'center', margin: 'var(--space-lg) 0' }}>
            <span style={{ color: 'var(--text-muted)' }}>or</span>
          </div>

          <div style={{ textAlign: 'center' }}>
            <button className="btn btn-primary" onClick={handleRandomAnalysis} id="demo-analyze-btn">
              Analyze Random Sample
            </button>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '8px' }}>
              Picks a random sample from the {groupLabel} dataset and runs the full pipeline
            </p>
          </div>
        </>
      )}

      {/* Loading */}
      {loading && (
        <div className="loading-spinner">
          <div className="spinner" />
          <p>Running HARUSPEX pipeline...</p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            {groupLabel} Model - Prediction - SHAP - Forensics - Correlation
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="error-message" style={{ marginBottom: 'var(--space-lg)' }}>
          {error}
        </div>
      )}

      {/* Results */}
      {report && <ReportView report={report} onReset={() => setReport(null)} />}
    </div>
  );
}

function ReportView({ report, onReset }: { report: PredictionReport; onReset: () => void }) {
  const isMALWARE = report.prediction.verdict === 'malware';
  const shapFeatures = report.xai.top_shap_features || [];
  const maxShap = Math.max(...shapFeatures.map(f => Math.abs(f.shap_value)), 0.01);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-lg)' }}>
        <h3>Investigation Report</h3>
        <button className="btn btn-secondary" onClick={onReset}>New Analysis</button>
      </div>

      {/* Verdict */}
      <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
        <div className="verdict-display">
          <div className={`verdict-label ${isMALWARE ? 'verdict-malware' : 'verdict-benign'}`}>
            {report.prediction.verdict}
          </div>
          <div className="verdict-confidence">
            Confidence: {(report.prediction.confidence * 100).toFixed(1)}% |
            Risk Score: {(report.prediction.risk_score * 100).toFixed(1)}% |
            Model: {report.prediction.model}
          </div>
        </div>
      </div>

      {/* Evidence summary */}
      <div className="card-grid" style={{ marginBottom: 'var(--space-lg)' }}>
        <div className="card stat-card">
          <div className="stat-value purple">{report.evidence_summary.total}</div>
          <div className="stat-label">Total Evidence</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value blue">{report.evidence_summary.correlated}</div>
          <div className="stat-label">Correlated</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value green">{report.evidence_summary.ai_only}</div>
          <div className="stat-label">AI Only</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value orange">{report.evidence_summary.forensic_only}</div>
          <div className="stat-label">Forensic Only</div>
        </div>
      </div>

      {/* SHAP Features */}
      {shapFeatures.length > 0 && (
        <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
          <div className="card-header">
            <div className="card-title">Top SHAP Feature Attributions</div>
          </div>
          <div className="shap-bar-container">
            {shapFeatures.slice(0, 15).map((feat: ShapFeature, i: number) => (
              <div className="shap-bar-row" key={i}>
                <div className="shap-bar-label" title={feat.feature_name}>{feat.feature_name}</div>
                <div className="shap-bar-track">
                  <div
                    className={`shap-bar-fill ${feat.shap_value >= 0 ? 'positive' : 'negative'}`}
                    style={{ width: `${(Math.abs(feat.shap_value) / maxShap) * 100}%` }}
                  />
                </div>
                <div className="shap-bar-value">{feat.shap_value >= 0 ? '+' : ''}{feat.shap_value.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evidence Items */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Evidence Items</div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {report.evidence_summary.correlated} correlated / {report.evidence_summary.total} total
          </span>
        </div>
        <div className="evidence-list">
          {report.evidence_items.map((item: EvidenceItem, i: number) => (
            <div className={`evidence-item ${item.source}`} key={i}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', minWidth: '110px' }}>
                {getSourceBadge(item.source)}
                <span className={`severity-badge ${getSeverityClass(item.severity)}`}>
                  {getSeverityLabel(item.severity)} ({(item.severity * 100).toFixed(0)}%)
                </span>
              </div>
              <div>
                <div className="evidence-indicator">{item.indicator}</div>
                {item.matched_forensic && (
                  <div className="evidence-matched">Forensic match: {item.matched_forensic}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sample info */}
      {report.sample?.sha256 && (
        <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
          <div className="card-header">
            <div className="card-title">Sample Information</div>
          </div>
          <table className="data-table">
            <tbody>
              <tr><td style={{ color: 'var(--text-secondary)', width: '150px' }}>SHA-256</td><td className="mono">{report.sample.sha256}</td></tr>
              <tr><td style={{ color: 'var(--text-secondary)' }}>Family</td><td>{report.sample.family || 'Unknown'}</td></tr>
              <tr><td style={{ color: 'var(--text-secondary)' }}>Category</td><td>{report.sample.category || 'Unknown'}</td></tr>
              {report.true_label && <tr><td style={{ color: 'var(--text-secondary)' }}>True Label</td><td>{report.true_label}</td></tr>}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}