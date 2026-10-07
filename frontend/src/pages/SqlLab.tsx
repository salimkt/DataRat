import { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart3, ChevronRight, Clock, Play, Table2 } from 'lucide-react';
import { api, type ChartType, type SqlResult } from '../api';
import { useApp } from '../context';
import { ChartView } from '../components/charts';
import { Card, DataTable, EmptyState, ErrorNote, Segmented, Spinner } from '../components/ui';

function defaultSql(id: string | null, cols: string[]): string {
  if (!id) return 'SELECT 1;';
  const select = cols.length ? cols.slice(0, 6).join(', ') : '*';
  return `SELECT ${select}\nFROM ${id}\nLIMIT 50;`;
}

export default function SqlLab() {
  const { selected, datasets, status } = useApp();
  const location = useLocation() as { state?: { sql?: string } };
  const [sql, setSql] = useState<string>(() => location.state?.sql ?? '');
  const [result, setResult] = useState<SqlResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [view, setView] = useState<'table' | 'chart'>('table');
  const [chartType, setChartType] = useState<ChartType>('bar');
  const ta = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!sql && selected) setSql(defaultSql(selected.id, selected.columns.map((c) => c.name)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.id]);

  const run = async () => {
    if (!sql.trim() || busy) return;
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    try {
      const r = await api.sql(sql, selected?.id);
      setResult(r);
      setElapsed(performance.now() - t0);
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const insert = (text: string) => {
    const el = ta.current;
    if (!el) {
      setSql((s) => s + text);
      return;
    }
    const { selectionStart: a, selectionEnd: b } = el;
    const next = sql.slice(0, a) + text + sql.slice(b);
    setSql(next);
    requestAnimationFrame(() => {
      el.focus();
      el.selectionStart = el.selectionEnd = a + text.length;
    });
  };

  return (
    <div className="sql-layout">
      <aside className="sql-schema" aria-label="Tables">
        <h2 className="section-title">Tables</h2>
        <p className="muted small">Every dataset is a table. Click to insert names.</p>
        {datasets.map((d) => (
          <details key={d.id} open={d.id === selected?.id} className="schema-table">
            <summary>
              <ChevronRight size={14} className="chev" aria-hidden="true" />
              <span className="mono">{d.id}</span>
              <span className="muted small">{d.rows.toLocaleString()}</span>
            </summary>
            <button className="schema-insert mono" onClick={() => insert(d.id)}>
              + insert table name
            </button>
            <ul>
              {d.columns.map((c) => (
                <li key={c.name}>
                  <button className="schema-col" onClick={() => insert(c.name)} title={`Insert ${c.name}`}>
                    <span className="mono">{c.name}</span>
                    <span className="muted small mono">{c.dtype}</span>
                  </button>
                </li>
              ))}
            </ul>
          </details>
        ))}
      </aside>

      <div className="stack">
        <Card
          title="Query editor"
          subtitle="DuckDB SQL · read-only (SELECT / WITH) · ⌘/Ctrl + Enter to run"
          actions={
            <button className="btn btn-primary" onClick={() => void run()} disabled={busy || status !== 'ok' || !sql.trim()}>
              {busy ? <Spinner size={16} label="Running…" /> : (
                <>
                  <Play size={16} aria-hidden="true" /> Run
                </>
              )}
            </button>
          }
        >
          <label htmlFor="sql-editor" className="sr-only">
            SQL query
          </label>
          <textarea
            id="sql-editor"
            ref={ta}
            className="sql-editor mono"
            value={sql}
            spellCheck={false}
            onChange={(e) => setSql(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                void run();
              } else if (e.key === 'Tab' && !e.shiftKey) {
                e.preventDefault();
                insert('  ');
              }
            }}
            rows={10}
          />
        </Card>

        {error && <ErrorNote message={error} />}

        {result ? (
          <Card
            title="Results"
            subtitle={
              <span className="row gap-sm">
                {result.rows.length.toLocaleString()} rows · {result.columns.length} columns
                {elapsed !== null && (
                  <>
                    <Clock size={12} aria-hidden="true" /> {Math.round(elapsed)} ms
                  </>
                )}
              </span>
            }
            actions={
              <div className="row gap-sm wrap">
                {view === 'chart' && (
                  <label className="inline-select">
                    <span className="sr-only">Chart type</span>
                    <select value={chartType} onChange={(e) => setChartType(e.target.value as ChartType)}>
                      <option value="bar">Bar</option>
                      <option value="line">Line</option>
                      <option value="area">Area</option>
                      <option value="pie">Pie</option>
                      <option value="scatter">Scatter</option>
                    </select>
                  </label>
                )}
                <Segmented
                  label="Result view"
                  value={view}
                  onChange={setView}
                  options={[
                    { value: 'table', label: <Table2 size={14} aria-label="Table" />, title: 'Table' },
                    { value: 'chart', label: <BarChart3 size={14} aria-label="Chart" />, title: 'Chart' },
                  ]}
                />
              </div>
            }
            pad={view === 'chart'}
          >
            {view === 'chart' ? (
              <ChartView spec={{ type: chartType, title: 'Query result' }} columns={result.columns} rows={result.rows} />
            ) : (
              <DataTable columns={result.columns} rows={result.rows} maxRows={1000} maxHeight={520} caption="Query results" />
            )}
          </Card>
        ) : (
          !error && (
            <EmptyState icon={<Play size={24} />} title="Run a query to see results">
              <p>Join across datasets freely — all loaded datasets are queryable by their table name.</p>
            </EmptyState>
          )
        )}
      </div>
    </div>
  );
}
