import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Calendar, Database, Hash, Key, Plug, Tag, Trash2, Type } from 'lucide-react';
import { api, type ColumnProfile, type Dataset, type Semantic } from '../api';
import { useApp, useAsync } from '../context';
import { Gauge, Heatmap, Histogram } from '../components/charts';
import { Badge, Card, DataTable, EmptyState, ErrorNote, Segmented, Skeleton, Spinner } from '../components/ui';
import { fmtCell, fmtNumber, fmtPercent, humanize, isNum } from '../format';

const SEM_ICON: Record<Semantic, typeof Hash> = { metric: Hash, dimension: Tag, date: Calendar, id: Key, text: Type };
const SEM_TONE: Record<Semantic, 'accent' | 'good' | 'warn' | 'neutral'> = {
  metric: 'accent',
  dimension: 'good',
  date: 'warn',
  id: 'neutral',
  text: 'neutral',
};

function SemanticBadge({ s }: { s: Semantic }) {
  const Icon = SEM_ICON[s] ?? Type;
  return (
    <Badge tone={SEM_TONE[s] ?? 'neutral'}>
      <Icon size={11} aria-hidden="true" /> {s}
    </Badge>
  );
}

function stat(v: unknown) {
  return isNum(v) ? fmtNumber(v, { compact: Math.abs(v) >= 100_000 }) : fmtCell(v);
}

function ColumnCard({ c }: { c: ColumnProfile }) {
  const top = (c.top_values ?? []).slice(0, 5);
  const maxTop = Math.max(1, ...top.map((t) => t.count));
  const allUnique = top.length > 0 && maxTop <= 1;
  return (
    <div className="card col-card">
      <div className="col-card-head">
        <div className="col-name" title={c.name}>
          {c.name}
        </div>
        <SemanticBadge s={c.semantic} />
      </div>
      <div className="col-meta muted small">
        <span className="mono">{c.dtype}</span>
        <span>{fmtNumber(c.unique)} unique</span>
        <span className={c.missing_pct > 5 ? 'text-bad' : ''}>{fmtPercent(c.missing_pct, { ratio: false })} missing</span>
      </div>
      {c.histogram && c.histogram.length > 0 ? (
        <Histogram bins={c.histogram} name={c.name} />
      ) : allUnique ? (
        <p className="muted small">Every value is unique — looks like an identifier.</p>
      ) : top.length > 0 ? (
        <ul className="topvals">
          {top.map((t, i) => (
            <li key={i}>
              <span className="tv-label" title={String(t.value)}>
                {fmtCell(t.value)}
              </span>
              <span className="tv-bar">
                <i style={{ width: `${(t.count / maxTop) * 100}%` }} />
              </span>
              <span className="tv-count">{fmtNumber(t.count)}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {(isNum(c.mean) || c.min !== undefined) && (
        <dl className="col-stats">
          {c.min !== undefined && c.min !== null && (
            <div>
              <dt>Min</dt>
              <dd>{stat(c.min)}</dd>
            </div>
          )}
          {isNum(c.median) && (
            <div>
              <dt>Median</dt>
              <dd>{stat(c.median)}</dd>
            </div>
          )}
          {isNum(c.mean) && (
            <div>
              <dt>Mean</dt>
              <dd>{stat(c.mean)}</dd>
            </div>
          )}
          {c.max !== undefined && c.max !== null && (
            <div>
              <dt>Max</dt>
              <dd>{stat(c.max)}</dd>
            </div>
          )}
        </dl>
      )}
    </div>
  );
}

function ProfileView({ ds }: { ds: Dataset }) {
  const { data, error, loading, reload } = useAsync(() => api.profile(ds.id), [ds.id]);
  if (error) return <ErrorNote message={error} onRetry={reload} />;
  if (loading || !data)
    return (
      <div className="stack">
        <Skeleton h={180} />
        <Skeleton h={240} />
      </div>
    );
  const p = data;
  // SPEC: *_pct fields are percentages on a 0-100 scale.
  const missingStr = fmtPercent(p.missing_cells_pct, { ratio: false });
  return (
    <div className="stack">
      <div className="profile-top">
        <Card title="Data quality" className="quality-card">
          <Gauge value={p.quality_score} />
          <p className="muted small center">Based on completeness, duplicates and type consistency.</p>
        </Card>
        <div className="grid-stats">
          <div className="card stat">
            <div className="stat-label">Rows</div>
            <div className="stat-value">{fmtNumber(p.rows)}</div>
          </div>
          <div className="card stat">
            <div className="stat-label">Columns</div>
            <div className="stat-value">{fmtNumber(p.columns)}</div>
          </div>
          <div className="card stat">
            <div className="stat-label">Missing cells</div>
            <div className="stat-value">{missingStr}</div>
          </div>
          <div className="card stat">
            <div className="stat-label">Duplicate rows</div>
            <div className="stat-value">{fmtNumber(p.duplicate_rows)}</div>
          </div>
        </div>
      </div>

      <section className="stack-sm" aria-labelledby="col-h">
        <h3 id="col-h" className="section-title">
          Columns
        </h3>
        <div className="grid-cols">
          {p.column_profiles.map((c) => (
            <ColumnCard key={c.name} c={c} />
          ))}
        </div>
      </section>

      {p.correlations?.columns?.length > 1 && (
        <Card title="Correlation matrix" subtitle="Pearson correlation between numeric columns — blue is positive, red is negative">
          <Heatmap columns={p.correlations.columns} matrix={p.correlations.matrix} />
        </Card>
      )}
    </div>
  );
}

function PreviewView({ ds }: { ds: Dataset }) {
  const { data, error, loading, reload } = useAsync(() => api.preview(ds.id, 100), [ds.id]);
  if (error) return <ErrorNote message={error} onRetry={reload} />;
  if (loading || !data) return <Skeleton h={360} />;
  return (
    <Card title="First 100 rows" pad={false}>
      <DataTable columns={data.columns} rows={data.rows} maxHeight={560} caption={`Preview of ${ds.name}`} />
    </Card>
  );
}

function SchemaView({ ds }: { ds: Dataset }) {
  return (
    <Card title="Schema" subtitle={`Table name in SQL: ${ds.id}`} pad={false}>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Column</th>
              <th scope="col">Type</th>
              <th scope="col">Role</th>
            </tr>
          </thead>
          <tbody>
            {ds.columns.map((c) => (
              <tr key={c.name}>
                <td className="mono">{c.name}</td>
                <td className="mono muted">{c.dtype}</td>
                <td>
                  <SemanticBadge s={c.semantic} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

type Tab = 'profile' | 'preview' | 'schema';

export default function Datasets() {
  const { datasets, selected, setSelectedId, refreshDatasets, toast, status, datasetsLoading } = useApp();
  const [tab, setTab] = useState<Tab>('profile');
  const [deleting, setDeleting] = useState<string | null>(null);

  const remove = async (d: Dataset) => {
    if (!window.confirm(`Remove “${d.name}” from the analysis server? This only unloads it from memory.`)) return;
    setDeleting(d.id);
    try {
      await api.deleteDataset(d.id);
      await refreshDatasets();
      toast(`Removed “${d.name}”`, 'success');
    } catch (e) {
      toast((e as Error).message, 'error');
    } finally {
      setDeleting(null);
    }
  };

  if (status === 'ok' && !datasetsLoading && datasets.length === 0) {
    return (
      <EmptyState icon={<Database size={28} />} title="No datasets loaded">
        <p>Connect a file, database or API — or reload one of the sample datasets.</p>
        <Link to="/connect" className="btn btn-primary">
          <Plug size={16} aria-hidden="true" /> Connect data
        </Link>
      </EmptyState>
    );
  }

  return (
    <div className="datasets-layout">
      <aside className="ds-list" aria-label="Datasets">
        <div className="row between">
          <h2 className="section-title">Datasets</h2>
          <Link to="/connect" className="btn btn-sm">
            <Plug size={14} aria-hidden="true" /> Add
          </Link>
        </div>
        {status !== 'ok' && [0, 1, 2].map((i) => <Skeleton key={i} h={64} />)}
        <ul>
          {datasets.map((d) => (
            <li key={d.id}>
              <div className={`ds-item ${selected?.id === d.id ? 'active' : ''}`}>
                <button className="ds-item-main" onClick={() => setSelectedId(d.id)} aria-current={selected?.id === d.id ? 'true' : undefined}>
                  <span className="ds-item-name">{d.name}</span>
                  <span className="muted small">
                    {d.source} · {d.rows.toLocaleString()} rows · {d.columns.length} cols
                  </span>
                </button>
                <button className="icon-btn danger" onClick={() => void remove(d)} aria-label={`Remove ${d.name}`} disabled={deleting === d.id}>
                  {deleting === d.id ? <Spinner size={14} /> : <Trash2 size={15} />}
                </button>
              </div>
            </li>
          ))}
        </ul>
      </aside>
      <div className="ds-detail stack">
        {selected ? (
          <>
            <div className="page-head">
              <div>
                <div className="eyebrow">Dataset</div>
                <h2 className="page-title">{selected.name}</h2>
                <div className="muted small">
                  {humanize(selected.source)} · <span className="mono">{selected.id}</span>
                </div>
              </div>
              <Segmented<Tab>
                label="Dataset view"
                value={tab}
                onChange={setTab}
                options={[
                  { value: 'profile', label: 'Profile' },
                  { value: 'preview', label: 'Preview' },
                  { value: 'schema', label: 'Schema' },
                ]}
              />
            </div>
            {tab === 'profile' && <ProfileView key={selected.id} ds={selected} />}
            {tab === 'preview' && <PreviewView key={selected.id} ds={selected} />}
            {tab === 'schema' && <SchemaView ds={selected} />}
          </>
        ) : (
          status === 'ok' && <p className="muted">Select a dataset.</p>
        )}
      </div>
    </div>
  );
}
