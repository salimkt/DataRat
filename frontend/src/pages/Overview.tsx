import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDownRight, ArrowUpRight, Bot, FileText, Lightbulb, Minus, Plug, Sparkles, TrendingDown, TrendingUp, X, Zap } from 'lucide-react';
import { api, type Dataset, type Kpi } from '../api';
import { useAsync } from '../context';
import { DatasetGate } from '../components/Gate';
import { BreakdownBars, Sparkline, TrendForecastChart } from '../components/charts';
import { Badge, Card, EmptyState, ErrorNote, InlineMarkdown, Segmented, Skeleton } from '../components/ui';
import { fmtByName, fmtChange, fmtDate, fmtKpi, humanize, isNum } from '../format';

const INTRO_KEY = 'datarat.introDismissed';

function IntroCard() {
  const [hidden, setHidden] = useState(() => {
    try {
      return localStorage.getItem(INTRO_KEY) === '1';
    } catch {
      return false;
    }
  });
  if (hidden) return null;
  const dismiss = () => {
    setHidden(true);
    try {
      localStorage.setItem(INTRO_KEY, '1');
    } catch {
      /* ignore */
    }
  };
  return (
    <section className="intro" aria-label="About DataRat">
      <button className="icon-btn intro-close" onClick={dismiss} aria-label="Dismiss introduction">
        <X size={16} />
      </button>
      <div className="intro-eyebrow">
        <Sparkles size={14} aria-hidden="true" /> AI decision intelligence
      </div>
      <h2>From raw data to board-ready answers in seconds.</h2>
      <p>
        DataRat connects to your spreadsheets, databases and APIs, profiles the data automatically, surfaces KPIs, anomalies and forecasts — and lets
        you ask questions in plain English. A team of AI agents plans the analysis, writes the SQL, picks the chart and briefs you like an analyst would.
      </p>
      <div className="intro-steps">
        <div>
          <Plug size={16} aria-hidden="true" />
          <span>
            <strong>Connect</strong> CSV, Excel, Sheets, Postgres, APIs…
          </span>
        </div>
        <div>
          <Zap size={16} aria-hidden="true" />
          <span>
            <strong>Auto-insights</strong> KPIs, anomalies, 30-day forecast
          </span>
        </div>
        <div>
          <Bot size={16} aria-hidden="true" />
          <span>
            <strong>Ask AI</strong> Planner → SQL → Analyst → Narrator
          </span>
        </div>
      </div>
      <div className="row gap-sm wrap">
        <Link to="/ask" className="btn btn-primary">
          <Sparkles size={16} aria-hidden="true" /> Ask a question
        </Link>
        <Link to="/connect" className="btn">
          <Plug size={16} aria-hidden="true" /> Connect your data
        </Link>
      </div>
    </section>
  );
}

const LOWER_IS_BETTER = /(cost|wait|duration|churn|latency|refund|complaint|discount|spend|time)/i;

function KpiCard({ kpi }: { kpi: Kpi }) {
  const ch = kpi.change_pct;
  const hasCh = isNum(ch);
  const inverted = LOWER_IS_BETTER.test(kpi.label);
  const good = hasCh && (inverted ? ch < 0 : ch > 0);
  const flat = hasCh && Math.abs(ch) < 0.05;
  const tone = !hasCh || flat ? 'neutral' : good ? 'good' : 'bad';
  const Icon = !hasCh || flat ? Minus : ch > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <div className="card kpi">
      <div className="kpi-label">{kpi.label}</div>
      <div className="kpi-value">{fmtKpi(kpi.value, kpi.format)}</div>
      <div className="kpi-meta">
        {hasCh ? (
          <span className={`chip chip-${tone}`}>
            <Icon size={13} aria-hidden="true" />
            {fmtChange(ch)}
          </span>
        ) : null}
        <span className="muted small">{kpi.period}</span>
      </div>
      <Sparkline data={kpi.sparkline} tone={tone === 'neutral' ? 'accent' : tone} />
    </div>
  );
}

function DatasetHeader({ ds }: { ds: Dataset }) {
  const metrics = ds.columns.filter((c) => c.semantic === 'metric').length;
  const dims = ds.columns.filter((c) => c.semantic === 'dimension').length;
  return (
    <div className="page-head">
      <div>
        <div className="eyebrow">Executive dashboard</div>
        <h2 className="page-title">{ds.name}</h2>
        <div className="row gap-sm wrap muted small">
          <Badge>{ds.source}</Badge>
          <span>{ds.rows.toLocaleString()} rows</span>
          <span>·</span>
          <span>{ds.columns.length} columns</span>
          <span>·</span>
          <span>
            {metrics} metrics, {dims} dimensions
          </span>
        </div>
      </div>
      <div className="row gap-sm wrap">
        <Link to="/brief" className="btn">
          <FileText size={16} aria-hidden="true" /> Executive brief
        </Link>
        <Link to="/ask" className="btn btn-primary">
          <Sparkles size={16} aria-hidden="true" /> Ask AI
        </Link>
      </div>
    </div>
  );
}

function Dashboard({ ds }: { ds: Dataset }) {
  const { data, error, loading, reload } = useAsync(() => api.insights(ds.id), [ds.id]);
  const [range, setRange] = useState<'forecast' | 'all'>('forecast');

  if (error) return <ErrorNote message={`Could not compute insights: ${error}`} onRetry={reload} />;
  if (loading || !data) {
    return (
      <div className="stack" aria-busy="true">
        <div className="grid-kpi">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="card card-body">
              <Skeleton h={12} w="45%" />
              <div style={{ height: 12 }} />
              <Skeleton h={28} w="60%" />
              <div style={{ height: 14 }} />
              <Skeleton h={34} />
            </div>
          ))}
        </div>
        <div className="grid-2-1">
          <div className="card card-body">
            <Skeleton h={300} />
          </div>
          <div className="card card-body">
            <Skeleton h={300} />
          </div>
        </div>
      </div>
    );
  }

  const ins = data;
  const showAll = range === 'all' || !ins.forecast;
  const history = (showAll ? ins.trend?.series : ins.forecast?.history) ?? ins.trend?.series ?? [];
  const smooth = showAll && history.length > 60;
  // Anomalies may be detected on an aggregated (e.g. weekly) series where `date` is the period start.
  // Only plot them on the daily chart when the value matches that day; otherwise label them as a period.
  const seriesByDate = new Map((ins.trend?.series ?? []).map((p) => [p.date, p.value]));
  const isPointAnomaly = (a: { date: string; value: number }) => {
    const v = seriesByDate.get(a.date);
    return v !== undefined && Math.abs(v - a.value) <= Math.max(1e-6, Math.abs(a.value) * 0.01);
  };
  const anomalyDates = ins.anomalies.filter(isPointAnomaly).map((a) => a.date);
  const visibleAnomalies = anomalyDates.filter((d) => history.some((h) => h.date === d));
  const metric = ins.forecast?.metric ?? ins.trend?.metric ?? '';

  return (
    <div className="stack">
      {ins.kpis.length > 0 && (
        <div className="grid-kpi">
          {ins.kpis.slice(0, 8).map((k) => (
            <KpiCard key={k.label} kpi={k} />
          ))}
        </div>
      )}

      <div className="grid-2-1">
        <Card
          title={metric ? `${humanize(metric)} trend` : 'Trend'}
          subtitle={
            !showAll && ins.forecast
              ? `Recent actuals with ${ins.forecast.forecast.length}-period forecast · ${ins.forecast.method}`
              : ins.trend
                ? `Full history by ${humanize(ins.trend.date_column)}${smooth ? ' · 7-day moving average' : ''}`
                : undefined
          }
          actions={
            ins.forecast && ins.trend ? (
              <Segmented
                label="Trend range"
                value={range}
                onChange={setRange}
                options={[
                  { value: 'forecast', label: 'Forecast' },
                  { value: 'all', label: 'All time' },
                ]}
              />
            ) : null
          }
        >
          <div className="legend-inline trend-legend" aria-hidden="true">
            <span>
              <i className="lg-line" /> {smooth ? 'Daily' : 'Actual'}
            </span>
            {smooth ? (
              <span>
                <i className="lg-line thick" /> 7-day avg
              </span>
            ) : showAll ? null : (
              <>
                <span>
                  <i className="lg-dash" /> Forecast
                </span>
                <span>
                  <i className="lg-band" /> Forecast range
                </span>
              </>
            )}
            {visibleAnomalies.length > 0 && (
              <span>
                <i className="lg-dot" /> Anomaly
              </span>
            )}
          </div>
          {history.length > 1 ? (
            <TrendForecastChart
              key={range}
              history={history}
              forecast={showAll ? undefined : ins.forecast?.forecast}
              metric={metric}
              anomalies={anomalyDates}
              smooth={smooth}
            />
          ) : (
            <EmptyState icon={<TrendingUp size={24} />} title="No time dimension detected">
              <p>Add a date column to unlock trends and forecasts.</p>
            </EmptyState>
          )}
        </Card>

        <Card title="Key findings" subtitle="Computed automatically from the data">
          {ins.highlights.length ? (
            <ul className="highlights">
              {ins.highlights.map((h, i) => (
                <li key={i}>
                  <Lightbulb size={16} aria-hidden="true" />
                  <span>
                    <InlineMarkdown>{h}</InlineMarkdown>
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No notable findings yet.</p>
          )}
        </Card>
      </div>

      {ins.breakdowns.length > 0 && (
        <div className="grid-3">
          {ins.breakdowns.slice(0, 3).map((b) => (
            <Card key={`${b.dimension}-${b.metric}`} title={`${humanize(b.metric)} by ${humanize(b.dimension)}`}>
              <BreakdownBars data={b.data} metric={b.metric} />
            </Card>
          ))}
        </div>
      )}

      <Card title="Anomalies" subtitle={ins.anomalies.length ? `${ins.anomalies.length} unusual data points detected (z-score)` : 'Statistical outliers versus expected values'}>
        {ins.anomalies.length === 0 ? (
          <p className="muted">No significant anomalies detected — the metric is behaving within expected ranges.</p>
        ) : (
          <ul className="anomalies">
            {ins.anomalies.slice(0, 8).map((a, i) => {
              const delta = a.expected ? ((a.value - a.expected) / Math.abs(a.expected)) * 100 : null;
              return (
                <li key={i} className="anomaly">
                  <span className={`anomaly-icon ${a.direction}`}>{a.direction === 'spike' ? <TrendingUp size={16} /> : <TrendingDown size={16} />}</span>
                  <div className="anomaly-main">
                    <div>
                      <strong>{a.direction === 'spike' ? 'Spike' : 'Dip'}</strong> in {humanize(a.metric)} {isPointAnomaly(a) ? 'on' : 'in the week of'}{' '}
                      {fmtDate(a.date, 'long')}
                    </div>
                    <div className="muted small">
                      {fmtByName(a.metric, a.value)} vs expected {fmtByName(a.metric, a.expected)}
                      {isNum(delta) ? ` (${fmtChange(delta)})` : ''}
                    </div>
                  </div>
                  <Badge tone={a.direction === 'spike' ? 'accent' : 'warn'}>z = {a.zscore.toFixed(1)}</Badge>
                </li>
              );
            })}
          </ul>
        )}
      </Card>
    </div>
  );
}

export default function Overview() {
  return (
    <div className="stack">
      <IntroCard />
      <DatasetGate>
        {(ds) => (
          <>
            <DatasetHeader ds={ds} />
            <Dashboard key={ds.id} ds={ds} />
          </>
        )}
      </DatasetGate>
    </div>
  );
}

