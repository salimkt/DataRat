import { useEffect, useState, type ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  CheckCircle2,
  Code2,
  Database,
  FileText,
  Github,
  LayoutDashboard,
  Menu,
  Moon,
  Plug,
  RefreshCw,
  Server,
  Settings,
  Sparkles,
  Sun,
  X,
  XCircle,
  Info,
  CloudOff,
} from 'lucide-react';
import { useApp } from '../context';
import { DEFAULT_API_BASE, getEnvApiBase, getStoredApiBase, llmLabel, setApiBase } from '../api';
import { Spinner } from './ui';

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/ask', label: 'Ask AI', icon: Sparkles },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/connect', label: 'Connect', icon: Plug },
  { to: '/sql', label: 'SQL Lab', icon: Code2 },
  { to: '/brief', label: 'Executive Brief', icon: FileText },
];

export function Logo() {
  return (
    <div className="logo">
      <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="8" fill="var(--accent)" />
        <path d="M8 22V14M13 22V9M18 22v-6M23 22V12" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div>
        <div className="logo-name">DataRat</div>
        <div className="logo-tag">Decision Intelligence</div>
      </div>
    </div>
  );
}

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { health, status } = useApp();
  return (
    <>
      <div className={`sidebar-scrim ${open ? 'show' : ''}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar ${open ? 'open' : ''}`} aria-label="Primary">
        <div className="sidebar-top">
          <Logo />
          <button className="icon-btn only-mobile" onClick={onClose} aria-label="Close navigation">
            <X size={18} />
          </button>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={onClose}>
              <n.icon size={18} aria-hidden="true" />
              <span>{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="engine-card">
            <div className="engine-row">
              <span className={`dot dot-${status === 'ok' ? 'good' : status === 'down' ? 'bad' : 'warn'}`} />
              <span>{status === 'ok' ? 'Server online' : status === 'down' ? 'Server offline' : 'Connecting…'}</span>
            </div>
            {health && (
              <div className="engine-row muted small">
                <Sparkles size={13} aria-hidden="true" />
                <span>{health.llm ? `${llmLabel(health)} AI agents active` : 'Rule-based agents (no LLM key)'}</span>
              </div>
            )}
            {health?.version && <div className="muted small">API v{health.version}</div>}
          </div>
        </div>
      </aside>
    </>
  );
}

function DatasetSwitcher() {
  const { datasets, selectedId, setSelectedId, status } = useApp();
  return (
    <label className="ds-switch">
      <Database size={16} aria-hidden="true" />
      <select aria-label="Active dataset" value={selectedId ?? ''} onChange={(e) => setSelectedId(e.target.value)} disabled={status !== 'ok' || datasets.length === 0}>
        {datasets.length === 0 && <option value="">{status === 'ok' ? 'No datasets' : 'Loading…'}</option>}
        {datasets.map((d) => (
          <option key={d.id} value={d.id}>
            {d.name} · {d.rows.toLocaleString()} rows
          </option>
        ))}
      </select>
    </label>
  );
}

function ServerBanner() {
  const { status, wakeElapsed, retryServer, apiBase, setSettingsOpen } = useApp();
  if (status === 'waking') {
    return (
      <div className="banner banner-warn" role="status" aria-live="polite">
        <Spinner size={16} />
        <div>
          <strong>Waking up the analysis server</strong> (Render free tier, ~30–50s)… <span className="muted">{wakeElapsed}s elapsed · retrying every 5s</span>
        </div>
      </div>
    );
  }
  if (status === 'down') {
    return (
      <div className="banner banner-bad" role="alert">
        <CloudOff size={16} aria-hidden="true" />
        <div>
          <strong>Can’t reach the analysis server</strong> at <code>{apiBase}</code>. It may still be starting or the URL may be wrong.
        </div>
        <div className="banner-actions">
          <button className="btn btn-sm" onClick={retryServer}>
            <RefreshCw size={14} aria-hidden="true" /> Retry
          </button>
          <button className="btn btn-sm btn-ghost" onClick={() => setSettingsOpen(true)}>
            <Settings size={14} aria-hidden="true" /> API settings
          </button>
        </div>
      </div>
    );
  }
  return null;
}

function SettingsDrawer() {
  const { settingsOpen, setSettingsOpen, reloadApiBase, toast, health, status } = useApp();
  const [value, setValue] = useState('');
  useEffect(() => {
    if (settingsOpen) setValue(getStoredApiBase() ?? getEnvApiBase());
  }, [settingsOpen]);
  useEffect(() => {
    if (!settingsOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setSettingsOpen(false);
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [settingsOpen, setSettingsOpen]);

  const save = () => {
    try {
      const u = new URL(value.trim());
      if (!/^https?:$/.test(u.protocol)) throw new Error();
    } catch {
      toast('Enter a valid http(s) URL', 'error');
      return;
    }
    setApiBase(value.trim() === getEnvApiBase() ? null : value);
    reloadApiBase();
    toast('API endpoint updated — reconnecting…', 'success');
    setSettingsOpen(false);
  };
  const reset = () => {
    setApiBase(null);
    setValue(getEnvApiBase());
    reloadApiBase();
    toast('Reset to default API endpoint', 'info');
  };

  return (
    <>
      <div className={`drawer-scrim ${settingsOpen ? 'show' : ''}`} onClick={() => setSettingsOpen(false)} aria-hidden="true" />
      <aside className={`drawer ${settingsOpen ? 'open' : ''}`} role="dialog" aria-modal="true" aria-label="Settings" aria-hidden={!settingsOpen}>
        <header className="drawer-head">
          <h2>Settings</h2>
          <button className="icon-btn" onClick={() => setSettingsOpen(false)} aria-label="Close settings" tabIndex={settingsOpen ? 0 : -1}>
            <X size={18} />
          </button>
        </header>
        <div className="drawer-body">
          <div className="field">
            <label htmlFor="api-base">Analysis server URL</label>
            <input
              id="api-base"
              type="url"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder={DEFAULT_API_BASE}
              tabIndex={settingsOpen ? 0 : -1}
              onKeyDown={(e) => e.key === 'Enter' && save()}
            />
            <p className="hint">
              The FastAPI backend that runs DuckDB and the AI agents. Default: <code>{getEnvApiBase()}</code>. Saved in this browser only.
            </p>
          </div>
          <div className="row gap-sm">
            <button className="btn btn-primary" onClick={save} tabIndex={settingsOpen ? 0 : -1}>
              Save & reconnect
            </button>
            <button className="btn btn-ghost" onClick={reset} tabIndex={settingsOpen ? 0 : -1}>
              Reset
            </button>
          </div>
          <div className="drawer-status">
            <h3>Connection</h3>
            <div className="engine-row">
              {status === 'ok' ? <CheckCircle2 size={16} className="text-good" /> : status === 'down' ? <XCircle size={16} className="text-bad" /> : <Spinner size={16} />}
              <span>{status === 'ok' ? 'Connected' : status === 'down' ? 'Unreachable' : 'Connecting…'}</span>
            </div>
            {health && (
              <>
                <div className="engine-row">
                  <Server size={16} aria-hidden="true" /> API version {health.version}
                </div>
                <div className="engine-row">
                  <Sparkles size={16} aria-hidden="true" /> {health.llm ? `LLM agents enabled (${llmLabel(health)}${health.model ? ` · ${health.model}` : ''})` : 'LLM disabled — deterministic rule-based agents'}
                </div>
              </>
            )}
          </div>
          <div className="drawer-about">
            <Info size={16} aria-hidden="true" />
            <p>
              DataRat is an open-source portfolio project by{' '}
              <a href="https://github.com/salimkt" target="_blank" rel="noreferrer" tabIndex={settingsOpen ? 0 : -1}>
                Muhammed Salim K T
              </a>
              . Your data is processed in-memory on the analysis server and is not persisted.
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}

function Toasts() {
  const { toasts, dismissToast } = useApp();
  return (
    <div className="toasts" aria-live="polite" aria-atomic="false">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`} role={t.kind === 'error' ? 'alert' : 'status'}>
          {t.kind === 'success' ? <CheckCircle2 size={16} /> : t.kind === 'error' ? <XCircle size={16} /> : <Info size={16} />}
          <span>{t.message}</span>
          <button className="icon-btn" onClick={() => dismissToast(t.id)} aria-label="Dismiss notification">
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

const TITLES: Record<string, string> = {
  '/': 'Overview',
  '/ask': 'Ask AI',
  '/datasets': 'Datasets',
  '/connect': 'Connect data',
  '/sql': 'SQL Lab',
  '/brief': 'Executive Brief',
};

export function Layout({ children }: { children: ReactNode }) {
  const { theme, toggleTheme, setSettingsOpen } = useApp();
  const [navOpen, setNavOpen] = useState(false);
  const loc = useLocation();
  const title = TITLES[loc.pathname] ?? 'DataRat';

  useEffect(() => {
    document.title = `${title} · DataRat`;
    window.scrollTo(0, 0);
  }, [title]);

  return (
    <div className="app">
      <a href="#main" className="skip-link" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus(); }}>
        Skip to content
      </a>
      <Sidebar open={navOpen} onClose={() => setNavOpen(false)} />
      <div className="main-col">
        <header className="topbar">
          <button className="icon-btn only-mobile" onClick={() => setNavOpen(true)} aria-label="Open navigation">
            <Menu size={20} />
          </button>
          <h1 className="topbar-title">{title}</h1>
          <div className="topbar-right">
            <DatasetSwitcher />
            <button className="icon-btn" onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title="Toggle theme">
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button className="icon-btn" onClick={() => setSettingsOpen(true)} aria-label="Open settings" title="Settings">
              <Settings size={18} />
            </button>
          </div>
        </header>
        <ServerBanner />
        <main id="main" className="content" tabIndex={-1}>
          {children}
        </main>
        <footer className="footer">
          <span>
            DataRat · AI decision intelligence · Built by{' '}
            <a href="https://github.com/salimkt" target="_blank" rel="noreferrer">
              <Github size={13} aria-hidden="true" /> Muhammed Salim K T
            </a>
          </span>
          <span className="muted">React · DuckDB · FastAPI · Claude</span>
        </footer>
      </div>
      <SettingsDrawer />
      <Toasts />
    </div>
  );
}
