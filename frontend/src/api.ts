// Typed API client for the DataRat backend. Types mirror SPEC.md exactly.

export const DEFAULT_API_BASE = 'https://datarat-api.onrender.com';
const STORAGE_KEY = 'datarat.apiBase';

// ---------- Types ----------

export type Semantic = 'metric' | 'dimension' | 'date' | 'id' | 'text';

export interface ColumnInfo {
  name: string;
  dtype: string;
  semantic: Semantic;
}

export interface Dataset {
  id: string;
  name: string;
  source: string;
  rows: number;
  columns: ColumnInfo[];
  created_at?: string;
}

export interface Health {
  status: 'ok' | string;
  llm: boolean;
  version: string;
}

export type FieldType = 'text' | 'password' | 'textarea' | 'select';

/** `options` is not strictly typed in SPEC; accept plain strings or {value,label} objects. */
export type FieldOption = string | { value: string; label?: string };

export interface ConnectorField {
  key: string;
  label: string;
  type: FieldType;
  placeholder?: string;
  options?: FieldOption[];
  required: boolean;
}

export type ConnectorCategory = 'file' | 'database' | 'cloud' | 'api';

export interface Connector {
  type: string;
  name: string;
  category: ConnectorCategory;
  description: string;
  icon: string;
  upload: boolean;
  fields: ConnectorField[];
}

export interface Preview {
  columns: string[];
  rows: unknown[][];
}

export interface HistogramBin {
  bin: string | number;
  count: number;
}

export interface TopValue {
  value: unknown;
  count: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  semantic: Semantic;
  missing_pct: number;
  unique: number;
  min?: number | string | null;
  max?: number | string | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  top_values?: TopValue[] | null;
  histogram?: HistogramBin[] | null;
}

export interface Profile {
  rows: number;
  columns: number;
  missing_cells_pct: number;
  duplicate_rows: number;
  quality_score: number;
  column_profiles: ColumnProfile[];
  correlations: { columns: string[]; matrix: number[][] };
}

export type KpiFormat = 'currency' | 'number' | 'percent';

export interface Kpi {
  label: string;
  value: number;
  format: KpiFormat;
  change_pct: number | null;
  period: string;
  sparkline: number[];
}

export interface SeriesPoint {
  date: string;
  value: number;
}

export interface ForecastPoint {
  date: string;
  value: number;
  lower: number;
  upper: number;
}

export interface Breakdown {
  dimension: string;
  metric: string;
  data: { label: string; value: number }[];
}

export interface Anomaly {
  date: string;
  metric: string;
  value: number;
  expected: number;
  zscore: number;
  direction: 'spike' | 'dip';
}

export interface Insights {
  kpis: Kpi[];
  trend: { date_column: string; metric: string; series: SeriesPoint[] } | null;
  breakdowns: Breakdown[];
  anomalies: Anomaly[];
  forecast: {
    metric: string;
    history: SeriesPoint[];
    forecast: ForecastPoint[];
    method: string;
  } | null;
  highlights: string[];
}

export interface AgentStep {
  agent: string;
  title: string;
  detail: string;
  duration_ms: number;
}

export type ChartType = 'bar' | 'line' | 'area' | 'pie' | 'scatter' | 'table' | 'kpi';

export interface ChartSpec {
  type: ChartType;
  x?: string;
  y?: string[];
  title: string;
}

export interface AskResult {
  question: string;
  steps: AgentStep[];
  sql: string;
  columns: string[];
  rows: unknown[][];
  chart: ChartSpec;
  narrative: string;
  recommendations: string[];
  follow_ups: string[];
}

export interface Brief {
  title: string;
  generated_at: string;
  summary: string;
  sections: { heading: string; body: string }[];
  kpis: Kpi[];
}

export interface SqlResult {
  columns: string[];
  rows: unknown[][];
}

// ---------- Base URL ----------

function clean(url: string): string {
  return url.trim().replace(/\/+$/, '');
}

export function getStoredApiBase(): string | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v && v.trim() ? clean(v) : null;
  } catch {
    return null;
  }
}

export function getEnvApiBase(): string {
  const env = import.meta.env.VITE_API_URL;
  return env && env.trim() ? clean(env) : DEFAULT_API_BASE;
}

export function getApiBase(): string {
  return getStoredApiBase() ?? getEnvApiBase();
}

export function setApiBase(url: string | null): void {
  try {
    if (url && url.trim()) localStorage.setItem(STORAGE_KEY, clean(url));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable */
  }
}

// ---------- Request core ----------

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function extractDetail(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d)))
      .join('; ');
  }
  const message = (body as { message?: unknown }).message;
  return typeof message === 'string' ? message : null;
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 120_000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${getApiBase()}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw new ApiError('The server took too long to respond.', 0);
    throw new ApiError('The analysis server could not be reached or failed while handling the request. Check your connection or the API URL in Settings, then try again.', 0);
  } finally {
    window.clearTimeout(timer);
  }
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!res.ok) {
    const msg = extractDetail(body) ?? (typeof body === 'string' && body ? body.slice(0, 300) : `Request failed (${res.status})`);
    throw new ApiError(msg, res.status);
  }
  return body as T;
}

function json(method: string, payload: unknown): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
}

const enc = encodeURIComponent;

// ---------- Endpoints ----------

export const api = {
  health: (timeoutMs = 8000) => request<Health>('/api/health', {}, timeoutMs),
  connectors: () => request<Connector[]>('/api/connectors'),
  datasets: () => request<Dataset[]>('/api/datasets'),
  dataset: (id: string) => request<Dataset>(`/api/datasets/${enc(id)}`),
  deleteDataset: (id: string) => request<{ ok: boolean }>(`/api/datasets/${enc(id)}`, { method: 'DELETE' }),
  preview: (id: string, limit = 100) => request<Preview>(`/api/datasets/${enc(id)}/preview?limit=${limit}`),
  profile: (id: string) => request<Profile>(`/api/datasets/${enc(id)}/profile`),
  insights: (id: string) => request<Insights>(`/api/datasets/${enc(id)}/insights`),
  connect: (type: string, fields: Record<string, string>, file?: File | null, name?: string) => {
    const fd = new FormData();
    fd.append('type', type);
    if (name && name.trim()) fd.append('name', name.trim());
    if (file) fd.append('file', file);
    for (const [k, v] of Object.entries(fields)) {
      if (v !== undefined && v !== null && String(v).trim() !== '') fd.append(k, v);
    }
    return request<Dataset>('/api/datasets/connect', { method: 'POST', body: fd }, 180_000);
  },
  ask: (dataset_id: string, question: string) => request<AskResult>('/api/ask', json('POST', { dataset_id, question }), 180_000),
  brief: (dataset_id: string) => request<Brief>('/api/brief', json('POST', { dataset_id }), 180_000),
  sql: (sql: string, dataset_id?: string) =>
    request<SqlResult>('/api/sql', json('POST', dataset_id ? { dataset_id, sql } : { sql }), 120_000),
};
