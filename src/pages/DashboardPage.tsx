import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import api, { type ModelInfo, type DatasetInfo } from '../services/api';

const CHART_COLORS = ['#00d4ff', '#00ff88', '#ff4466', '#ff8844', '#aa66ff', '#ffcc00'];

export default function DashboardPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [models, dataset] = await Promise.all([
          api.getModels(),
          api.getDatasetInfo(),
        ]);
        setModelInfo(models);
        setDatasetInfo(dataset);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data. Is the API running?');
      }
    };
    load();
  }, []);

  if (error) {
    return (
      <div>
        <div className="page-header">
          <h2>📊 Model Dashboard</h2>
          <p>Model performance metrics and dataset overview</p>
        </div>
        <div className="error-message">⚠️ {error}</div>
        <div style={{ color: 'var(--text-muted)', marginTop: 'var(--space-md)', fontSize: '0.85rem' }}>
          Start the API server with: <code style={{ color: 'var(--accent-cyan)' }}>cd backend && source .venv/bin/activate && python -m haruspex.api.main</code>
        </div>
      </div>
    );
  }

  // Category pie chart data
  const categoryData = datasetInfo
    ? Object.entries(datasetInfo.categories)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([name, value]) => ({ name, value }))
    : [];

  // Top families bar chart data (real per-family sample counts from the API)
  const familyData = datasetInfo?.top_families ?? [];

  return (
    <div>
      <div className="page-header">
        <h2>📊 Model Dashboard</h2>
        <p>Model performance metrics and dataset overview</p>
      </div>

      {/* Dataset stats */}
      {datasetInfo && (
        <div className="card-grid" style={{ marginBottom: 'var(--space-lg)' }}>
          <div className="card stat-card">
            <div className="stat-value cyan">{datasetInfo.total_samples.toLocaleString()}</div>
            <div className="stat-label">Total Samples</div>
          </div>
          <div className="card stat-card">
            <div className="stat-value green">{datasetInfo.n_benign.toLocaleString()}</div>
            <div className="stat-label">Benign</div>
          </div>
          <div className="card stat-card">
            <div className="stat-value red">{datasetInfo.n_malware.toLocaleString()}</div>
            <div className="stat-label">Malware</div>
          </div>
          <div className="card stat-card">
            <div className="stat-value orange">{datasetInfo.n_families}</div>
            <div className="stat-label">Families</div>
          </div>
        </div>
      )}

      {/* Model info */}
      {modelInfo && (
        <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
          <div className="card-header">
            <div className="card-title">🤖 Loaded Models</div>
          </div>
          {modelInfo.loaded.length === 0 ? (
            <div className="empty-state" style={{ padding: 'var(--space-lg)' }}>
              <p>No models loaded. Run <code style={{ color: 'var(--accent-cyan)' }}>python scripts/train.py</code> first.</p>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Default</th>
                </tr>
              </thead>
              <tbody>
                {modelInfo.loaded.map((name) => (
                  <tr key={name}>
                    <td style={{ fontWeight: 600 }}>{name.toUpperCase()}</td>
                    <td><span className="severity-badge severity-low">LOADED</span></td>
                    <td>{name === modelInfo.default ? '✓' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        {/* Category distribution */}
        {categoryData.length > 0 && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">Malware Categories</div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={categoryData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label={({ name, percent }) => `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`}
                  labelLine={false}
                >
                  {categoryData.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#1a2235', border: '1px solid #2a3a55', borderRadius: '8px' }}
                  labelStyle={{ color: '#e8edf5' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Label distribution */}
        {datasetInfo && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">Label Distribution</div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { name: 'Benign', count: datasetInfo.n_benign },
                { name: 'Malware', count: datasetInfo.n_malware },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3e" />
                <XAxis dataKey="name" stroke="#8b95a8" />
                <YAxis stroke="#8b95a8" />
                <Tooltip
                  contentStyle={{ background: '#1a2235', border: '1px solid #2a3a55', borderRadius: '8px' }}
                  labelStyle={{ color: '#e8edf5' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  <Cell fill="#00ff88" />
                  <Cell fill="#ff4466" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Top families */}
      {familyData.length > 0 && (
        <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
          <div className="card-header">
            <div className="card-title">Top Malware Families</div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={familyData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3e" />
              <XAxis type="number" stroke="#8b95a8" />
              <YAxis type="category" dataKey="name" stroke="#8b95a8" width={120} />
              <Tooltip
                contentStyle={{ background: '#1a2235', border: '1px solid #2a3a55', borderRadius: '8px' }}
                labelStyle={{ color: '#e8edf5' }}
              />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {familyData.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature info */}
      {datasetInfo && (
        <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
          <div className="card-header">
            <div className="card-title">📐 Feature Space</div>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            {datasetInfo.n_features.toLocaleString()} EMBER-compatible features across 9 groups:
            Byte Histogram (256), Byte Entropy (256), String Info (104),
            General Info (10), Header Info (62), Section Info (255),
            Import Info (1280), Export Info (128), Data Directories (30)
          </p>
        </div>
      )}
    </div>
  );
}
