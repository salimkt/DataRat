import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { api, getApiBase, type Dataset, type Health } from './api';

export type ServerStatus = 'checking' | 'waking' | 'ok' | 'down';
export type Theme = 'light' | 'dark';

export interface Toast {
  id: number;
  kind: 'success' | 'error' | 'info';
  message: string;
}

interface AppState {
  status: ServerStatus;
  health: Health | null;
  wakeElapsed: number;
  retryServer: () => void;
  apiBase: string;
  reloadApiBase: () => void;

  datasets: Dataset[];
  datasetsLoading: boolean;
  datasetsError: string | null;
  refreshDatasets: () => Promise<Dataset[]>;
  selectedId: string | null;
  selected: Dataset | null;
  setSelectedId: (id: string) => void;

  theme: Theme;
  toggleTheme: () => void;

  settingsOpen: boolean;
  setSettingsOpen: (open: boolean) => void;

  toasts: Toast[];
  toast: (message: string, kind?: Toast['kind']) => void;
  dismissToast: (id: number) => void;
}

const Ctx = createContext<AppState | null>(null);

const SELECTED_KEY = 'datarat.dataset';
const THEME_KEY = 'datarat.theme';
const WAKE_BANNER_DELAY = 2500;
const RETRY_EVERY = 5000;
const MAX_WAIT = 90_000;

function readLS(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function writeLS(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

export function AppProvider({ children }: { children: ReactNode }) {
  // ----- theme -----
  const [theme, setTheme] = useState<Theme>(() => {
    const attr = document.documentElement.getAttribute('data-theme');
    return attr === 'dark' ? 'dark' : 'light';
  });
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    writeLS(THEME_KEY, theme);
  }, [theme]);
  const toggleTheme = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), []);

  // ----- toasts -----
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const dismissToast = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);
  const toast = useCallback(
    (message: string, kind: Toast['kind'] = 'info') => {
      const id = nextId.current++;
      setToasts((ts) => [...ts.slice(-3), { id, kind, message }]);
      window.setTimeout(() => dismissToast(id), kind === 'error' ? 7000 : 4500);
    },
    [dismissToast],
  );

  // ----- server health with cold-start handling -----
  const [apiBase, setApiBaseState] = useState(getApiBase());
  const [status, setStatus] = useState<ServerStatus>('checking');
  const [health, setHealth] = useState<Health | null>(null);
  const [wakeElapsed, setWakeElapsed] = useState(0);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const start = Date.now();
    setStatus('checking');
    setWakeElapsed(0);
    const bannerTimer = window.setTimeout(() => {
      if (!cancelled) setStatus((s) => (s === 'checking' ? 'waking' : s));
    }, WAKE_BANNER_DELAY);
    const tick = window.setInterval(() => {
      if (!cancelled) setWakeElapsed(Math.round((Date.now() - start) / 1000));
    }, 1000);

    (async () => {
      while (!cancelled) {
        try {
          const h = await api.health(12_000);
          if (cancelled) return;
          setHealth(h);
          setStatus('ok');
          return;
        } catch {
          if (cancelled) return;
          if (Date.now() - start >= MAX_WAIT) {
            setStatus('down');
            return;
          }
          setStatus('waking');
          await sleep(RETRY_EVERY);
        }
      }
    })().finally(() => {
      window.clearTimeout(bannerTimer);
      window.clearInterval(tick);
    });

    return () => {
      cancelled = true;
      window.clearTimeout(bannerTimer);
      window.clearInterval(tick);
    };
  }, [attempt, apiBase]);

  const retryServer = useCallback(() => setAttempt((a) => a + 1), []);
  const reloadApiBase = useCallback(() => {
    setApiBaseState(getApiBase());
    setAttempt((a) => a + 1);
  }, []);

  // ----- datasets -----
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [selectedId, setSelectedIdState] = useState<string | null>(() => readLS(SELECTED_KEY));

  const setSelectedId = useCallback((id: string) => {
    setSelectedIdState(id);
    writeLS(SELECTED_KEY, id);
  }, []);

  const refreshDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    setDatasetsError(null);
    try {
      const list = await api.datasets();
      setDatasets(list);
      setSelectedIdState((cur) => {
        if (cur && list.some((d) => d.id === cur)) return cur;
        const fallback = list.find((d) => d.id === 'sales')?.id ?? list[0]?.id ?? null;
        if (fallback) writeLS(SELECTED_KEY, fallback);
        return fallback;
      });
      return list;
    } catch (e) {
      setDatasetsError((e as Error).message);
      return [];
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (status === 'ok') void refreshDatasets();
  }, [status, refreshDatasets]);

  const selected = useMemo(() => datasets.find((d) => d.id === selectedId) ?? null, [datasets, selectedId]);

  const [settingsOpen, setSettingsOpen] = useState(false);

  const value: AppState = {
    status,
    health,
    wakeElapsed,
    retryServer,
    apiBase,
    reloadApiBase,
    datasets,
    datasetsLoading,
    datasetsError,
    refreshDatasets,
    selectedId,
    selected,
    setSelectedId,
    theme,
    toggleTheme,
    settingsOpen,
    setSettingsOpen,
    toasts,
    toast,
    dismissToast,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp must be used inside <AppProvider>');
  return v;
}

/** Small async-data hook keyed by deps. */
export function useAsync<T>(fn: (() => Promise<T>) | null, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!fn) {
      setData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, error, loading, reload: () => setNonce((n) => n + 1), setData };
}
