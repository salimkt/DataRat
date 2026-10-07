import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  type TooltipProps,
} from 'recharts';
import type { ChartSpec, ForecastPoint, HistogramBin, SeriesPoint } from '../api';
import { useApp } from '../context';
import { fmtAxis, fmtByName, fmtCell, fmtDate, fmtDateTick, fmtNumber, humanize, isNum, looksLikeDate } from '../format';
import { DataTable } from './ui';

// ---------- Theme-aware colors (read from CSS variables) ----------

export interface ChartColors {
  series: string[];
  grid: string;
  axis: string;
  text: string;
  muted: string;
  surface: string;
  good: string;
  bad: string;
  band: string;
}

function readColors(): ChartColors {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string, fb: string) => s.getPropertyValue(n).trim() || fb;
  return {
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v(`--series-${i}`, '#2a78d6')),
    grid: v('--chart-grid', '#e7e6e2'),
    axis: v('--text-muted', '#8a8983'),
    text: v('--text', '#111'),
    muted: v('--text-muted', '#777'),
    surface: v('--surface', '#fff'),
    good: v('--good', '#0e8a4f'),
    bad: v('--bad', '#d23f3f'),
    band: v('--series-1', '#2a78d6'),
  };
}

export function useChartColors(): ChartColors {
  const { theme } = useApp();
  const [c, setC] = useState<ChartColors>(readColors);
  useEffect(() => {
    // wait a frame so the data-theme attribute has applied
    const id = requestAnimationFrame(() => setC(readColors()));
    return () => cancelAnimationFrame(id);
  }, [theme]);
  return c;
}

// ---------- Tooltip ----------

function ChartTooltip({
  active,
  payload,
  label,
  xName,
  hint = (k) => k,
}: TooltipProps<number, string> & { xName?: string; hint?: (k: string) => string }) {
  if (!active || !payload || payload.length === 0) return null;
  const lbl = looksLikeDate(label) ? fmtDate(label, 'long') : fmtCell(label);
  return (
    <div className="chart-tip">
      {label !== undefined && label !== '' && (
        <div className="chart-tip-label">
          {xName ? <span className="muted">{humanize(xName)}: </span> : null}
          {lbl}
        </div>
      )}
      {payload.map((p, i) => {
        const raw = p.value as unknown;
        const val = Array.isArray(raw)
          ? `${fmtByName(hint(String(p.dataKey ?? p.name)), raw[0])} – ${fmtByName(hint(String(p.dataKey ?? p.name)), raw[1])}`
          : fmtByName(hint(String(p.name ?? p.dataKey)), raw);
        return (
          <div key={i} className="chart-tip-row">
            <span className="chart-tip-swatch" style={{ background: (p.color as string) || (p.payload?.fill as string) }} />
            <span className="chart-tip-name">{humanize(String(p.name ?? p.dataKey))}</span>
            <span className="chart-tip-val">{val}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------- Sparkline ----------

export function Sparkline({ data, tone = 'accent', height = 36 }: { data: number[]; tone?: 'accent' | 'good' | 'bad'; height?: number }) {
  const c = useChartColors();
  const color = tone === 'good' ? c.good : tone === 'bad' ? c.bad : c.series[0];
  const pts = useMemo(() => (data || []).filter(isNum).map((v, i) => ({ i, v })), [data]);
  const id = useMemo(() => `spark-${Math.random().toString(36).slice(2, 8)}`, []);
  if (pts.length < 2) return <div style={{ height }} />;
  return (
    <div style={{ height }} aria-hidden="true">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={pts} margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.25} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis hide domain={['dataMin', 'dataMax']} />
          <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.75} fill={`url(#${id})`} isAnimationActive={false} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------- Trend + forecast ----------

export function TrendForecastChart({
  history,
  forecast,
  metric,
  anomalies = [],
  height = 300,
  smooth = false,
}: {
  history: SeriesPoint[];
  forecast?: ForecastPoint[];
  metric: string;
  anomalies?: string[];
  height?: number;
  /** add a 7-period moving average and fade the raw series */
  smooth?: boolean;
}) {
  const c = useChartColors();
  const data = useMemo(() => {
    const rows: Record<string, unknown>[] = history.map((p, i) => {
      const row: Record<string, unknown> = { date: p.date, actual: p.value };
      if (smooth && i >= 6) {
        let sum = 0;
        for (let k = i - 6; k <= i; k++) sum += history[k].value;
        row.avg = sum / 7;
      }
      return row;
    });
    const fc = forecast ?? [];
    if (fc.length && rows.length) {
      // connect the forecast line to the last actual point
      const last = rows[rows.length - 1];
      last.forecast = last.actual;
      last.band = [last.actual, last.actual];
    }
    for (const f of fc) rows.push({ date: f.date, forecast: f.value, band: [f.lower, f.upper] });
    return rows;
  }, [history, forecast, smooth]);
  const spanDays = useMemo(() => {
    const first = data[0]?.date as string | undefined;
    const last = data[data.length - 1]?.date as string | undefined;
    if (!first || !last) return 0;
    return (new Date(last).getTime() - new Date(first).getTime()) / 86_400_000;
  }, [data]);
  const tickFmt = (v: unknown) => (spanDays > 0 && spanDays < 200 ? fmtDate(v, 'short') : fmtDateTick(v));
  const anomalySet = useMemo(() => new Set(anomalies), [anomalies]);
  const hasFc = (forecast?.length ?? 0) > 0;

  return (
    <div style={{ height }} role="img" aria-label={`${humanize(metric)} over time${hasFc ? ' with 30-period forecast' : ''}`}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="date" tickFormatter={tickFmt} tick={{ fill: c.axis, fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={40} />
          <YAxis tickFormatter={(v) => fmtAxis(metric, v)} tick={{ fill: c.axis, fontSize: 11 }} axisLine={false} tickLine={false} width={56} />
          <Tooltip content={<ChartTooltip />} />
          {hasFc && <Area dataKey="band" name="Forecast range" stroke="none" fill={c.series[0]} fillOpacity={0.14} isAnimationActive={false} activeDot={false} />}
          <Line
            type="monotone"
            dataKey="actual"
            name={humanize(metric)}
            stroke={c.series[0]}
            strokeWidth={smooth ? 1 : 2}
            strokeOpacity={smooth ? 0.35 : 1}
            isAnimationActive={false}
            dot={(props: { cx?: number; cy?: number; payload?: { date?: string }; index?: number }) => {
              const { cx, cy, payload, index } = props;
              if (!payload?.date || !anomalySet.has(payload.date) || cx === undefined || cy === undefined)
                return <g key={`d${index}`} />;
              return <circle key={`d${index}`} cx={cx} cy={cy} r={5} fill={c.series[1]} stroke={c.surface} strokeWidth={2} />;
            }}
            activeDot={{ r: 4, strokeWidth: 2, stroke: c.surface }}
          />
          {smooth && <Line type="monotone" dataKey="avg" name="7-day average" stroke={c.series[0]} strokeWidth={2} dot={false} isAnimationActive={false} />}
          {hasFc && (
            <Line type="monotone" dataKey="forecast" name="Forecast" stroke={c.series[0]} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------- Horizontal bar breakdown ----------

export function BreakdownBars({ data, metric, height }: { data: { label: string; value: number }[]; metric: string; height?: number }) {
  const c = useChartColors();
  const rows = data.slice(0, 10);
  const h = height ?? Math.max(160, rows.length * 34 + 20);
  return (
    <div style={{ height: h }} role="img" aria-label={`${humanize(metric)} breakdown`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }} barCategoryGap={6}>
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="label"
            width={110}
            tick={{ fill: c.muted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: string) => (String(v).length > 16 ? `${String(v).slice(0, 15)}…` : String(v))}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: c.grid, opacity: 0.5 }} />
          <Bar
            dataKey="value"
            name={humanize(metric)}
            fill={c.series[0]}
            radius={[0, 4, 4, 0]}
            maxBarSize={22}
            isAnimationActive={false}
            label={{ position: 'right', fill: c.muted, fontSize: 11, formatter: (v: number) => fmtAxis(metric, v) }}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------- Histogram ----------

export function Histogram({ bins, height = 90, name }: { bins: HistogramBin[]; height?: number; name: string }) {
  const c = useChartColors();
  const data = bins.map((b) => ({ bin: typeof b.bin === 'number' ? fmtNumber(b.bin) : String(b.bin), count: b.count }));
  return (
    <div style={{ height }} role="img" aria-label={`Distribution of ${name}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }} barCategoryGap={1}>
          <XAxis dataKey="bin" hide />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: c.grid, opacity: 0.5 }} />
          <Bar dataKey="count" name="Rows" fill={c.series[0]} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------- Quality gauge ----------

export function Gauge({ value, size = 168, label = 'Quality score' }: { value: number; size?: number; label?: string }) {
  const c = useChartColors();
  const v = Math.max(0, Math.min(100, isNum(value) ? value : 0));
  const r = size / 2 - 12;
  const cx = size / 2;
  const cy = size / 2;
  const start = Math.PI * 0.75;
  const sweep = Math.PI * 1.5;
  const arc = (frac: number) => {
    const a0 = start;
    const a1 = start + sweep * frac;
    const x0 = cx + r * Math.cos(a0);
    const y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy + r * Math.sin(a1);
    const large = sweep * frac > Math.PI ? 1 : 0;
    return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
  };
  const tone = v >= 85 ? c.good : v >= 65 ? 'var(--warn)' : c.bad;
  const verdict = v >= 85 ? 'Excellent' : v >= 65 ? 'Fair' : 'Needs attention';
  return (
    <div className="gauge" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v)} aria-label={label}>
      <svg width={size} height={size * 0.82} viewBox={`0 0 ${size} ${size * 0.82}`}>
        <path d={arc(1)} stroke={c.grid} strokeWidth={12} fill="none" strokeLinecap="round" />
        {v > 0 && <path d={arc(v / 100)} stroke={tone} strokeWidth={12} fill="none" strokeLinecap="round" />}
        <text x={cx} y={cy + 4} textAnchor="middle" fontSize={size * 0.22} fontWeight={700} fill={c.text}>
          {Math.round(v)}
        </text>
        <text x={cx} y={cy + 24} textAnchor="middle" fontSize={12} fill={c.muted}>
          {verdict}
        </text>
      </svg>
    </div>
  );
}

// ---------- Correlation heatmap (diverging blue ↔ red, gray midpoint) ----------

function mix(a: number[], b: number[], t: number) {
  return a.map((x, i) => Math.round(x + (b[i] - x) * t));
}
function hex(h: string): number[] {
  const m = h.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
}

export function Heatmap({ columns, matrix }: { columns: string[]; matrix: number[][] }) {
  const { theme } = useApp();
  const mid = hex(theme === 'dark' ? '#383835' : '#f0efec');
  const pos = hex(theme === 'dark' ? '#3987e5' : '#1c5cab');
  const neg = hex(theme === 'dark' ? '#e66767' : '#c43a39');
  const color = (v: number) => {
    if (!isNum(v)) return 'transparent';
    const t = Math.min(1, Math.abs(v));
    const rgb = mix(mid, v >= 0 ? pos : neg, t);
    return `rgb(${rgb.join(',')})`;
  };
  const ink = (v: number) => (Math.abs(v) > 0.55 ? '#fff' : 'var(--text)');
  const n = columns.length;
  return (
    <div className="heatmap-scroll">
      <div className="heatmap" style={{ gridTemplateColumns: `minmax(90px, max-content) repeat(${n}, minmax(44px, 1fr))` }} role="table" aria-label="Correlation matrix">
        <div role="row" style={{ display: 'contents' }}>
          <div className="hm-corner" />
          {columns.map((cName) => (
            <div key={cName} className="hm-col" role="columnheader" title={cName}>
              <span>{cName}</span>
            </div>
          ))}
        </div>
        {columns.map((rName, i) => (
          <div key={rName} role="row" style={{ display: 'contents' }}>
            <div className="hm-row" role="rowheader" title={rName}>
              {rName}
            </div>
            {columns.map((cName, j) => {
              const v = matrix[i]?.[j];
              return (
                <div
                  key={cName}
                  role="cell"
                  className="hm-cell"
                  style={{ background: color(v), color: ink(v) }}
                  title={`${rName} × ${cName}: ${isNum(v) ? v.toFixed(2) : 'n/a'}`}
                >
                  {isNum(v) ? v.toFixed(2) : '–'}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="hm-legend" aria-hidden="true">
        <span>-1</span>
        <span className="hm-legend-bar" style={{ background: `linear-gradient(90deg, ${color(-1)}, ${color(0)}, ${color(1)})` }} />
        <span>+1</span>
      </div>
    </div>
  );
}

// ---------- Generic chart from an AI chart spec ----------

const MAX_SERIES = 8;

function inferSpec(spec: ChartSpec, columns: string[], rows: unknown[][]) {
  const numeric = columns.filter((_, i) => rows.length > 0 && rows.slice(0, 50).every((r) => r[i] === null || isNum(r[i])) && rows.some((r) => isNum(r[i])));
  let x = spec.x && columns.includes(spec.x) ? spec.x : undefined;
  if (!x) x = columns.find((c) => !numeric.includes(c)) ?? columns[0];
  let y = (spec.y ?? []).filter((c) => columns.includes(c) && c !== x);
  if (y.length === 0) y = numeric.filter((c) => c !== x);
  return { x, y: y.slice(0, MAX_SERIES), numeric };
}

export function ChartView({ spec, columns, rows, height = 320 }: { spec: ChartSpec; columns: string[]; rows: unknown[][]; height?: number }) {
  const c = useChartColors();
  const { x, y } = useMemo(() => inferSpec(spec, columns, rows), [spec, columns, rows]);
  const data = useMemo(
    () =>
      rows.map((r) => {
        const o: Record<string, unknown> = {};
        columns.forEach((col, i) => (o[col] = r[i]));
        return o;
      }),
    [columns, rows],
  );

  let type = spec.type;
  if (type !== 'table' && type !== 'kpi' && (y.length === 0 || rows.length === 0)) type = 'table';
  if (type === 'kpi' && rows.length > 3) type = 'table';

  if (type === 'table') return <DataTable columns={columns} rows={rows} maxHeight={height + 60} caption={spec.title} />;

  if (type === 'kpi') {
    const r = rows[0] ?? [];
    const items = (y.length ? y : columns).slice(0, 4);
    return (
      <div className="kpi-result">
        {items.map((col) => {
          const v = r[columns.indexOf(col)];
          return (
            <div key={col} className="kpi-result-item">
              <div className="kpi-result-value">{fmtByName(col, v)}</div>
              <div className="kpi-result-label">{humanize(col)}</div>
            </div>
          );
        })}
      </div>
    );
  }

  const xIsDate = data.length > 0 && looksLikeDate(data[0][x]);
  // Wide results (one column per series, e.g. regions) carry the metric only in the title.
  const hint = (k: string) => (y.length > 1 ? `${spec.title || ''} ${k}` : k);
  const yName = hint(y[0]);
  const axisProps = { tick: { fill: c.axis, fontSize: 11 }, axisLine: false, tickLine: false } as const;
  const xTick = (v: unknown) => (xIsDate ? fmtDateTick(v) : String(v).length > 14 ? `${String(v).slice(0, 13)}…` : String(v));
  const showLegend = y.length >= 2;
  const aria = `${spec.title || 'Chart'}: ${y.map(humanize).join(', ')} by ${humanize(x)}`;

  if (type === 'pie') {
    const metric = y[0];
    const sorted = [...data].sort((a, b) => Number(b[metric] ?? 0) - Number(a[metric] ?? 0));
    const top = sorted.slice(0, 7).map((d) => ({ name: String(d[x]), value: Number(d[metric] ?? 0) }));
    const rest = sorted.slice(7).reduce((s, d) => s + Number(d[metric] ?? 0), 0);
    if (rest > 0) top.push({ name: 'Other', value: rest });
    return (
      <div style={{ height }} role="img" aria-label={aria}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Tooltip content={<ChartTooltip />} />
            <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ fontSize: 12, color: c.muted }} />
            <Pie data={top} dataKey="value" nameKey="name" innerRadius="52%" outerRadius="80%" paddingAngle={1.5} stroke={c.surface} strokeWidth={2} isAnimationActive={false}>
              {top.map((_, i) => (
                <Cell key={i} fill={c.series[i % MAX_SERIES]} name={top[i].name} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (type === 'scatter') {
    const yKey = y[0];
    const xKey = isNum(data[0]?.[x]) ? x : y[1] ?? x;
    const yk = xKey === x ? yKey : y[0];
    return (
      <div style={{ height }} role="img" aria-label={aria}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={c.grid} />
            <XAxis type="number" dataKey={xKey} name={humanize(xKey)} tickFormatter={(v) => fmtAxis(xKey, v)} {...axisProps} />
            <YAxis type="number" dataKey={yk} name={humanize(yk)} tickFormatter={(v) => fmtAxis(yk, v)} {...axisProps} width={56} />
            <ZAxis range={[60, 60]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v: number, n: string) => [fmtByName(n, v), humanize(n)]} contentStyle={{ background: c.surface, border: `1px solid ${c.grid}`, borderRadius: 8 }} />
            <Scatter data={data} fill={c.series[0]} fillOpacity={0.75} stroke={c.surface} strokeWidth={1} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (type === 'line' || type === 'area') {
    const Chart = type === 'area' ? AreaChart : LineChart;
    return (
      <div style={{ height }} role="img" aria-label={aria}>
        <ResponsiveContainer width="100%" height="100%">
          <Chart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={c.grid} vertical={false} />
            <XAxis dataKey={x} tickFormatter={xTick} {...axisProps} minTickGap={32} />
            <YAxis tickFormatter={(v) => fmtAxis(yName, v)} {...axisProps} width={56} />
            <Tooltip content={<ChartTooltip xName={x} hint={hint} />} />
            {showLegend && <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} formatter={(v: string) => humanize(v)} />}
            {y.map((k, i) =>
              type === 'area' ? (
                <Area key={k} type="monotone" dataKey={k} stroke={c.series[i]} strokeWidth={2} fill={c.series[i]} fillOpacity={y.length > 1 ? 0.08 : 0.15} isAnimationActive={false} dot={false} />
              ) : (
                <Line key={k} type="monotone" dataKey={k} stroke={c.series[i]} strokeWidth={2} dot={data.length <= 24 ? { r: 3 } : false} isAnimationActive={false} />
              ),
            )}
          </Chart>
        </ResponsiveContainer>
      </div>
    );
  }

  // bar
  const horizontal = !xIsDate && data.length <= 15 && y.length <= 2;
  if (horizontal) {
    const h = Math.max(height, data.length * (y.length > 1 ? 44 : 32) + 40);
    return (
      <div style={{ height: Math.min(h, 620) }} role="img" aria-label={aria}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 64, bottom: 4, left: 4 }} barCategoryGap={6} barGap={2}>
            <CartesianGrid stroke={c.grid} horizontal={false} />
            <XAxis type="number" tickFormatter={(v) => fmtAxis(yName, v)} {...axisProps} />
            <YAxis type="category" dataKey={x} width={120} tickFormatter={xTick} {...axisProps} tick={{ fill: c.muted, fontSize: 12 }} />
            <Tooltip content={<ChartTooltip xName={x} hint={hint} />} cursor={{ fill: c.grid, opacity: 0.5 }} />
            {showLegend && <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v: string) => humanize(v)} />}
            {y.map((k, i) => (
              <Bar
                key={k}
                dataKey={k}
                fill={c.series[i]}
                radius={[0, 4, 4, 0]}
                maxBarSize={24}
                isAnimationActive={false}
                label={y.length === 1 ? { position: 'right', fill: c.muted, fontSize: 11, formatter: (v: number) => fmtAxis(k, v) } : undefined}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }
  return (
    <div style={{ height }} role="img" aria-label={aria}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }} barGap={2}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey={x} tickFormatter={xTick} {...axisProps} minTickGap={12} />
          <YAxis tickFormatter={(v) => fmtAxis(yName, v)} {...axisProps} width={56} />
          <Tooltip content={<ChartTooltip xName={x} hint={hint} />} cursor={{ fill: c.grid, opacity: 0.5 }} />
          {showLegend && <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v: string) => humanize(v)} />}
          {y.map((k, i) => (
            <Bar key={k} dataKey={k} fill={c.series[i]} radius={[4, 4, 0, 0]} maxBarSize={40} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
