import { useState, useEffect } from 'react';

interface ResultEntry {
  id: number;
  timestamp: string;
  verdict: string;
  confidence: number;
  risk_score: number;
  evidence_count: number;
  correlated_count: number;
  sample_index?: number;
}

export default function ResultsPage() {
  const [results, setResults] = useState<ResultEntry[]>([]);

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem('haruspex_results') || '[]');
    setResults(stored);
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('haruspex_results');
    setResults([]);
  };

  return (
    <div>
      <div className="page-header">
        <h2>📋 Analysis Results</h2>
        <p>History of previous analyses from this session</p>
      </div>

      {results.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <p>No analysis results yet. Head to the Analyze page to get started.</p>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-lg)' }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              {results.length} results
            </span>
            <button className="btn btn-secondary" onClick={clearHistory} id="clear-history-btn">
              Clear History
            </button>
          </div>

          <div className="card">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Verdict</th>
                  <th>Confidence</th>
                  <th>Risk Score</th>
                  <th>Evidence</th>
                  <th>Correlated</th>
                  <th>Sample</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.id}>
                    <td className="mono" style={{ fontSize: '0.8rem' }}>
                      {new Date(r.timestamp).toLocaleString()}
                    </td>
                    <td>
                      <span className={`severity-badge ${r.verdict === 'malware' ? 'severity-critical' : 'severity-low'}`}>
                        {r.verdict.toUpperCase()}
                      </span>
                    </td>
                    <td className="mono">{(r.confidence * 100).toFixed(1)}%</td>
                    <td className="mono">{(r.risk_score * 100).toFixed(1)}%</td>
                    <td className="mono">{r.evidence_count}</td>
                    <td className="mono">{r.correlated_count}</td>
                    <td className="mono">{r.sample_index !== undefined ? `#${r.sample_index}` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
