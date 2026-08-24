import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { useState, useEffect } from 'react';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import ExplorerPage from './pages/ExplorerPage';
import api from './services/api';
import './index.css';

type ModelGroup = 'bodmas' | 'ember';

interface ModelGroupMeta {
  label: string;
  description: string;
  icon: string;
}

const MODEL_GROUPS: Record<ModelGroup, ModelGroupMeta> = {
  bodmas: {
    label: 'BODMAS Models',
    description: 'Trained on BODMAS dataset',
    icon: '🧬',
  },
  ember: {
    label: 'EMBER Models',
    description: 'Trained on EMBER 2018 dataset',
    icon: '🔥',
  },
};

function App() {
  const [apiStatus, setApiStatus] = useState<'online' | 'offline'>('offline');
  const [modelGroup, setModelGroup] = useState<ModelGroup>('bodmas');
  const [availableGroups, setAvailableGroups] = useState<string[]>([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.health();
        setApiStatus('online');
        // Discover available model groups from API
        if (health.model_groups) {
          setAvailableGroups(Object.keys(health.model_groups));
        }
      } catch {
        setApiStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const activeGroupMeta = MODEL_GROUPS[modelGroup];

  return (
    <BrowserRouter>
      <div className="app-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-logo">
              <div className="sidebar-logo-icon">H</div>
              <div>
                <h1>HARUSPEX</h1>
                <p>AI Malware Forensics</p>
              </div>
            </div>
          </div>

          {/* Model Group Selector */}
          <div className="model-group-selector">
            <div className="model-group-label">Inference Models</div>
            <div className="model-group-dropdown-wrapper">
              <button
                className="model-group-dropdown-trigger"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                id="model-group-selector"
              >
                <span className="model-group-icon">{activeGroupMeta.icon}</span>
                <div className="model-group-info">
                  <span className="model-group-name">{activeGroupMeta.label}</span>
                  <span className="model-group-desc">{activeGroupMeta.description}</span>
                </div>
                <span className={`model-group-chevron ${dropdownOpen ? 'open' : ''}`}>▾</span>
              </button>

              {dropdownOpen && (
                <div className="model-group-dropdown-menu">
                  {(Object.entries(MODEL_GROUPS) as [ModelGroup, ModelGroupMeta][]).map(
                    ([key, meta]) => {
                      const isAvailable = availableGroups.includes(key);
                      return (
                        <button
                          key={key}
                          className={`model-group-option ${modelGroup === key ? 'active' : ''} ${
                            !isAvailable ? 'disabled' : ''
                          }`}
                          onClick={() => {
                            if (isAvailable) {
                              setModelGroup(key);
                              setDropdownOpen(false);
                            }
                          }}
                          disabled={!isAvailable}
                        >
                          <span className="model-group-icon">{meta.icon}</span>
                          <div className="model-group-info">
                            <span className="model-group-name">{meta.label}</span>
                            <span className="model-group-desc">
                              {isAvailable ? meta.description : 'Not trained yet'}
                            </span>
                          </div>
                          {modelGroup === key && <span className="model-group-check">✓</span>}
                        </button>
                      );
                    },
                  )}
                </div>
              )}
            </div>
          </div>

          <nav className="sidebar-nav">
            <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">🔬</span>
              <span>Analyze</span>
            </NavLink>
            <NavLink to="/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">📊</span>
              <span>Dashboard</span>
            </NavLink>
            <NavLink to="/explorer" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">🔍</span>
              <span>Explorer</span>
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <div className="api-status">
              <div className={`status-dot ${apiStatus === 'offline' ? 'offline' : ''}`} />
              <span>API {apiStatus}</span>
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<UploadPage modelGroup={modelGroup} />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/explorer" element={<ExplorerPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
