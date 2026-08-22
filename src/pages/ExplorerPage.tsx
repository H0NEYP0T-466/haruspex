import { useState, useEffect, useCallback } from 'react';
import api, { type DatasetSample, type DatasetInfo, type PredictionReport, type EvidenceItem, type ShapFeature } from '../services/api';

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

export default function ExplorerPage() {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [totalSamples, setTotalSamples] = useState(0);
  const [offset, setOffset] = useState(0);
  const [filters, setFilters] = useState<{ family?: string; category?: string; label?: string }>({});
  const [selectedReport, setSelectedReport] = useState<PredictionReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDatasetInfo().then(setDatasetInfo).catch(() => setError('Failed to load dataset info'));
  }, []);

  const loadSamples = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.listSamples(filters, offset);
      setSamples(result.samples);
      setTotalSamples(result.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load samples');
    } finally {
      setLoading(false);
    }
  }, [filters, offset]);

  useEffect(() => { loadSamples(); }, [loadSamples]);

  const handleExplain = useCallback(async (sampleIndex: number) => {
    setExplaining(sampleIndex);
    setSelectedReport(null);
    try {
      const report = await api.explainSample(sampleIndex);
      setSelectedReport(report);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Explanation failed');
    } finally {
      setExplaining(null);
    }
  }, []);

  return (
    <div>
      <div className="page-header">
        <h2>🔍 Sample Explorer</h2>
        <p>Browse and analyze BODMAS dataset samples</p>
      </div>

      {error && <div className="error-message" style={{ marginBottom: 'var(--space-lg)' }}>⚠️ {error}</div>}

      {/* Filters */}
      <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
        <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            id="filter-label"
            value={filters.label || ''}
            onChange={(e) => { setFilters({ ...filters, label: e.target.value || undefined }); setOffset(0); }}
          >
            <option value="">All Labels</option>
            <option value="benign">Benign</option>
            <option value="malware">Malware</option>
          </select>

          <select
            id="filter-category"
            value={filters.category || ''}
            onChange={(e) => { setFilters({ ...filters, category: e.target.value || undefined }); setOffset(0); }}
          >
            <option value="">All Categories</option>
            {datasetInfo && Object.keys(datasetInfo.categories).sort().map((cat) => (
              <option key={cat} value={cat}>{cat} ({datasetInfo.categories[cat]})</option>
            ))}
          </select>

          <input
            type="text"
            id="filter-family"
            placeholder="Filter by family..."
            onChange={(e) => { setFilters({ ...filters, family: e.target.value || undefined }); setOffset(0); }}
            style={{ width: '200px' }}
          />

          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginLeft: 'auto' }}>
            {totalSamples.toLocaleString()} matches
          </span>
        </div>
      </div>

      {/* Samples table */}
      <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
        {loading ? (
          <div className="loading-spinner"><div className="spinner" /><p>Loading samples...</p></div>
        ) : (
          <>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>SHA-256</th>
                  <th>Label</th>
                  <th>Family</th>
                  <th>Category</th>
                  <th>Timestamp</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s) => (
                  <tr key={s.index}>
                    <td className="mono">{s.index}</td>
                    <td className="mono" style={{ fontSize: '0.75rem', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {s.sha256}
                    </td>
                    <td>
                      <span className={`severity-badge ${s.label === 'malware' ? 'severity-critical' : 'severity-low'}`}>
                        {s.label}
                      </span>
                    </td>
                    <td>{s.family || '—'}</td>
                    <td>{s.category}</td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>{s.timestamp.split('T')[0]}</td>
                    <td>
                      <button
                        className="btn btn-primary"
                        style={{ padding: '4px 12px', fontSize: '0.78rem' }}
                        onClick={() => handleExplain(s.index)}
                        disabled={explaining === s.index}
                        id={`explain-btn-${s.index}`}
                      >
                        {explaining === s.index ? '⏳' : '🔬'} Explain
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 'var(--space-md)' }}>
              <button
                className="btn btn-secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - 50))}
                id="prev-page-btn"
              >
                ← Previous
              </button>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Showing {offset + 1}–{Math.min(offset + 50, totalSamples)} of {totalSamples.toLocaleString()}
              </span>
              <button
                className="btn btn-secondary"
                disabled={offset + 50 >= totalSamples}
                onClick={() => setOffset(offset + 50)}
                id="next-page-btn"
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>

      {/* Selected sample explanation */}
      {selectedReport && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              Investigation Report — Sample #{selectedReport.sample_index}
            </div>
            <button className="btn btn-secondary" onClick={() => setSelectedReport(null)} style={{ padding: '4px 12px', fontSize: '0.78rem' }}>
              ✕ Close
            </button>
          </div>

          {/* Verdict */}
          <div className="verdict-display" style={{ padding: 'var(--space-md)' }}>
            <div className={`verdict-label ${selectedReport.prediction.verdict === 'malware' ? 'verdict-malware' : 'verdict-benign'}`} style={{ fontSize: '1.5rem' }}>
              {selectedReport.prediction.verdict}
            </div>
            <div className="verdict-confidence" style={{ fontSize: '0.9rem' }}>
              Confidence: {(selectedReport.prediction.confidence * 100).toFixed(1)}% |
              True label: {selectedReport.true_label || '—'} |
              Family: {selectedReport.sample?.family || '—'}
            </div>
          </div>

          {/* Evidence summary */}
          <div className="card-grid" style={{ marginBottom: 'var(--space-md)' }}>
            <div style={{ textAlign: 'center' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
                {selectedReport.evidence_summary.correlated}
              </span>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>CORRELATED</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {selectedReport.evidence_summary.ai_only}
              </span>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>AI ONLY</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-orange)' }}>
                {selectedReport.evidence_summary.forensic_only}
              </span>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FORENSIC</div>
            </div>
          </div>

          {/* Top SHAP features */}
          {selectedReport.xai.top_shap_features.length > 0 && (
            <div style={{ marginBottom: 'var(--space-md)' }}>
              <h4 style={{ marginBottom: 'var(--space-sm)', fontSize: '0.95rem' }}>Top SHAP Features</h4>
              <div className="shap-bar-container">
                {selectedReport.xai.top_shap_features.slice(0, 10).map((feat: ShapFeature, i: number) => {
                  const maxShap = Math.max(...selectedReport.xai.top_shap_features.map(f => Math.abs(f.shap_value)), 0.01);
                  return (
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
                  );
                })}
              </div>
            </div>
          )}

          {/* Evidence items (top 10) */}
          <h4 style={{ marginBottom: 'var(--space-sm)', fontSize: '0.95rem' }}>Evidence Items</h4>
          <div className="evidence-list">
            {selectedReport.evidence_items.slice(0, 10).map((item: EvidenceItem, i: number) => (
              <div className={`evidence-item ${item.source}`} key={i}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', minWidth: '80px' }}>
                  <span className={`severity-badge ${getSeverityClass(item.severity)}`} style={{ fontSize: '0.65rem' }}>
                    {getSeverityLabel(item.severity)}
                  </span>
                </div>
                <div className="evidence-indicator" style={{ fontSize: '0.82rem' }}>
                  {item.indicator}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
