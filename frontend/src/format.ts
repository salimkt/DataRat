import type { KpiFormat } from './api';

const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const integer = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** $1.2M, $840K, $12.50 */
export function fmtCurrency(v: number | null | undefined, opts: { compact?: boolean } = {}): string {
  if (!isNum(v)) return '—';
  const useCompact = opts.compact ?? Math.abs(v) >= 10_000;
  const sign = v < 0 ? '-' : '';
  const abs = Math.abs(v);
  if (useCompact) return `${sign}$${compact.format(abs)}`;
  return `${sign}$${abs.toLocaleString('en-US', { minimumFractionDigits: abs < 100 && abs % 1 !== 0 ? 2 : 0, maximumFractionDigits: 2 })}`;
}

/** 12.3K, 1.2M, 842, 3.14 */
export function fmtNumber(v: number | null | undefined, opts: { compact?: boolean } = {}): string {
  if (!isNum(v)) return '—';
  const useCompact = opts.compact ?? Math.abs(v) >= 10_000;
  if (useCompact) return compact.format(v);
  if (Number.isInteger(v)) return integer.format(v);
  return decimal.format(v);
}

/**
 * Percent. Values with |v| <= 1 are treated as ratios (0.123 → 12.3%) unless `ratio: false`.
 * KPI values with format "percent" are ambiguous in SPEC; we use the heuristic.
 */
export function fmtPercent(v: number | null | undefined, opts: { ratio?: boolean; digits?: number } = {}): string {
  if (!isNum(v)) return '—';
  const ratio = opts.ratio ?? Math.abs(v) <= 1;
  const pct = ratio ? v * 100 : v;
  const digits = opts.digits ?? 1;
  return `${pct.toFixed(digits)}%`;
}

/** Signed change for already-percent numbers: +12.4% / -3.0% */
export function fmtChange(v: number | null | undefined): string {
  if (!isNum(v)) return '—';
  const s = v > 0 ? '+' : '';
  return `${s}${v.toFixed(1)}%`;
}

export function fmtKpi(v: number, format: KpiFormat): string {
  if (format === 'currency') return fmtCurrency(v);
  if (format === 'percent') return fmtPercent(v, { ratio: false }); // KPI percents are on a 0-100 scale
  return fmtNumber(v);
}

const MONEY_RE = /(revenue|sales|amount|profit|cost|spend|price|income|gmv|arr|mrr|value|attributed)/i;
const PCT_RE = /(pct|percent|rate|ratio|share|margin|discount|_pct$)/i;

/** Guess a formatter from a column/metric name. */
export function fmtByName(name: string | undefined, v: unknown): string {
  if (!isNum(v)) return fmtCell(v);
  if (name && MONEY_RE.test(name) && !/count|units/i.test(name)) return fmtCurrency(v);
  if (name && PCT_RE.test(name) && Math.abs(v) <= 1) return fmtPercent(v);
  return fmtNumber(v);
}

/** Axis tick: always compact. */
export function fmtAxis(name: string | undefined, v: unknown): string {
  if (!isNum(v)) return String(v ?? '');
  if (name && MONEY_RE.test(name) && !/count|units/i.test(name)) return fmtCurrency(v, { compact: Math.abs(v) >= 1000 });
  return fmtNumber(v, { compact: Math.abs(v) >= 1000 });
}

/** Generic table cell. */
export function fmtCell(v: unknown): string {
  if (v === null || v === undefined) return '∅';
  if (typeof v === 'number') return Number.isInteger(v) ? integer.format(v) : decimal.format(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}/;

export function fmtDate(v: unknown, style: 'short' | 'long' = 'short'): string {
  if (typeof v !== 'string' || !ISO_DATE.test(v)) return String(v ?? '');
  const d = new Date(v.length === 10 ? `${v}T00:00:00` : v);
  if (Number.isNaN(d.getTime())) return v;
  return style === 'short'
    ? d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function fmtDateTick(v: unknown): string {
  if (typeof v !== 'string' || !ISO_DATE.test(v)) return String(v ?? '');
  const d = new Date(v.length === 10 ? `${v}T00:00:00` : v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
}

export function looksLikeDate(v: unknown): boolean {
  return typeof v === 'string' && ISO_DATE.test(v);
}

export function humanize(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function fmtMs(ms: number): string {
  if (!isNum(ms)) return '';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
