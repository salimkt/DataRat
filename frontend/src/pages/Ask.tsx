import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowUp,
  BarChart3,
  Brain,
  Check,
  ChevronDown,
  Code2,
  Copy,
  Database,
  ExternalLink,
  Lightbulb,
  MessageSquareText,
  RotateCcw,
  Sparkles,
  Table2,
  Target,
  type LucideIcon,
} from 'lucide-react';
import { api, type AskResult, type Dataset } from '../api';
import { useApp } from '../context';
import { DatasetGate } from '../components/Gate';
import { ChartView } from '../components/charts';
import { Card, DataTable, ErrorNote, InlineMarkdown, Markdown, Segmented } from '../components/ui';
import { fmtMs } from '../format';

// ---------- Agents ----------

const AGENTS: { key: string; name: string; icon: LucideIcon; working: string }[] = [
  { key: 'planner', name: 'Planner', icon: Target, working: 'Understanding intent and selecting relevant columns…' },
  { key: 'sql', name: 'SQL Engineer', icon: Database, working: 'Writing and executing DuckDB SQL…' },
  { key: 'analyst', name: 'Analyst', icon: BarChart3, working: 'Computing statistics and choosing a chart…' },
  { key: 'narrator', name: 'Narrator', icon: MessageSquareText, working: 'Drafting the executive summary…' },
];

function agentIcon(name: string): LucideIcon {
  const n = name.toLowerCase();
  if (n.includes('plan')) return Target;
  if (n.includes('sql') || n.includes('engineer')) return Database;
  if (n.includes('analy')) return BarChart3;
  if (n.includes('narr') || n.includes('writer')) return MessageSquareText;
  return Brain;
}

function LiveTimeline() {
  const [active, setActive] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setActive((a) => Math.min(a + 1, AGENTS.length - 1)), 700);
    return () => window.clearInterval(id);
  }, []);
  return (
    <ol className="timeline live" aria-live="polite" aria-label="Agent progress">
      {AGENTS.map((a, i) => {
        const state = i < active ? 'done' : i === active ? 'active' : 'pending';
        return (
          <li key={a.key} className={`tl-step ${state}`}>
            <span className="tl-icon">{state === 'done' ? <Check size={14} /> : <a.icon size={14} />}</span>
            <div className="tl-body">
              <div className="tl-title">{a.name}</div>
              <div className="tl-detail">{state === 'pending' ? 'Waiting' : state === 'done' ? 'Done' : a.working}</div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function StepsTimeline({ steps }: { steps: AskResult['steps'] }) {
  const total = steps.reduce((s, x) => s + (x.duration_ms || 0), 0);
  return (
    <details className="disclosure" open>
      <summary>
        <Sparkles size={14} aria-hidden="true" /> How the agents worked <span className="muted small">· {steps.length} steps · {fmtMs(total)}</span>
        <ChevronDown size={14} className="chev" aria-hidden="true" />
      </summary>
      <ol className="timeline">
        {steps.map((s, i) => {
          const Icon = agentIcon(s.agent);
          return (
            <li key={i} className="tl-step done">
              <span className="tl-icon">
                <Icon size={14} />
              </span>
              <div className="tl-body">
                <div className="tl-title">
                  {s.agent} <span className="muted">· {s.title}</span>
                  <span className="tl-time">{fmtMs(s.duration_ms)}</span>
                </div>
                {s.detail && <div className="tl-detail">{s.detail}</div>}
              </div>
            </li>
          );
        })}
      </ol>
    </details>
  );
}

// ---------- Suggested questions ----------

const PRESETS: Record<string, string[]> = {
  sales: [
    'What is total revenue by region?',
    'Show monthly revenue trend',
    'Top 5 products by profit',
    'Compare revenue by channel',
    'Average discount by customer segment',
    'What is revenue growth by month?',
  ],
  marketing: [
    'Total spend by channel',
    'Show monthly conversions trend',
    'Top channels by revenue attributed',
    'Average clicks by channel',
    'Compare leads by channel',
  ],
  support_calls: [
    'Average csat by queue',
    'Show monthly call volume trend',
    'Top 5 agents by resolved calls',
    'Average wait_sec by queue',
    'Average sentiment by topic',
  ],
};

function suggestFor(ds: Dataset): string[] {
  if (PRESETS[ds.id]) return PRESETS[ds.id];
  const metrics = ds.columns.filter((c) => c.semantic === 'metric').map((c) => c.name);
  const dims = ds.columns.filter((c) => c.semantic === 'dimension').map((c) => c.name);
  const date = ds.columns.find((c) => c.semantic === 'date')?.name;
  const m = metrics[0];
  const out: string[] = [];
  if (m && dims[0]) out.push(`Total ${m} by ${dims[0]}`);
  if (m && date) out.push(`Show ${m} over time by month`);
  if (m && dims[1]) out.push(`Top 5 ${dims[1]} by ${m}`);
  if (metrics[1] && dims[0]) out.push(`Average ${metrics[1]} by ${dims[0]}`);
  out.push('How many rows are there?');
  return out.slice(0, 5);
}

// ---------- Result card ----------

function CopyBtn({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      className="btn btn-ghost btn-sm"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setOk(true);
          window.setTimeout(() => setOk(false), 1500);
        });
      }}
    >
      {ok ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />} {ok ? 'Copied' : label}
    </button>
  );
}

function ResultView({ r, onAsk, datasetId }: { r: AskResult; onAsk: (q: string) => void; datasetId: string }) {
  const [view, setView] = useState<'chart' | 'table'>(r.chart.type === 'table' ? 'table' : 'chart');
  const navigate = useNavigate();
  const canChart = r.chart.type !== 'table' && r.rows.length > 0;
  return (
    <div className="answer stack-sm">
      <StepsTimeline steps={r.steps} />

      {r.narrative && (
        <div className="narrative">
          <Markdown>{r.narrative}</Markdown>
        </div>
      )}

      <Card
        title={r.chart.title || 'Result'}
        subtitle={`${r.rows.length.toLocaleString()} row${r.rows.length === 1 ? '' : 's'} · ${r.columns.length} column${r.columns.length === 1 ? '' : 's'}`}
        actions={
          canChart ? (
            <Segmented
              label="Result view"
              value={view}
              onChange={setView}
              options={[
                { value: 'chart', label: <BarChart3 size={14} aria-label="Chart" />, title: 'Chart' },
                { value: 'table', label: <Table2 size={14} aria-label="Table" />, title: 'Table' },
              ]}
            />
          ) : undefined
        }
      >
        {r.rows.length === 0 ? (
          <p className="muted">The query returned no rows.</p>
        ) : view === 'chart' && canChart ? (
          <ChartView spec={r.chart} columns={r.columns} rows={r.rows} />
        ) : (
          <DataTable columns={r.columns} rows={r.rows} maxHeight={380} caption={r.chart.title} />
        )}
      </Card>

      {r.sql && (
        <details className="disclosure">
          <summary>
            <Code2 size={14} aria-hidden="true" /> Generated SQL
            <ChevronDown size={14} className="chev" aria-hidden="true" />
          </summary>
          <div className="sql-block">
            <pre>
              <code>{r.sql}</code>
            </pre>
            <div className="row gap-sm">
              <CopyBtn text={r.sql} label="Copy SQL" />
              <button className="btn btn-ghost btn-sm" onClick={() => navigate('/sql', { state: { sql: r.sql, datasetId } })}>
                <ExternalLink size={14} aria-hidden="true" /> Open in SQL Lab
              </button>
            </div>
          </div>
        </details>
      )}

      {r.recommendations.length > 0 && (
        <div className="recs">
          <h4>
            <Lightbulb size={16} aria-hidden="true" /> Recommendations
          </h4>
          <ol>
            {r.recommendations.map((x, i) => (
              <li key={i}>
                <InlineMarkdown>{x}</InlineMarkdown>
              </li>
            ))}
          </ol>
        </div>
      )}

      {r.follow_ups.length > 0 && (
        <div className="chips" aria-label="Follow-up questions">
          {r.follow_ups.map((f) => (
            <button key={f} className="q-chip" onClick={() => onAsk(f)}>
              {f}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------- Page ----------

interface Turn {
  id: number;
  question: string;
  status: 'loading' | 'done' | 'error';
  result?: AskResult;
  error?: string;
}

// Keeps conversations alive while navigating between pages (in-memory only).
let threadCache: Record<string, Turn[]> = {};

function Chat({ ds }: { ds: Dataset }) {
  const { health } = useApp();
  const [threads, setThreads] = useState<Record<string, Turn[]>>(() => threadCache);
  useEffect(() => {
    threadCache = Object.fromEntries(Object.entries(threads).map(([k, v]) => [k, v.filter((t) => t.status !== 'loading')]));
  }, [threads]);
  const [input, setInput] = useState('');
  const nextId = useRef(1);
  const lastRef = useRef<HTMLElement>(null);
  const turns = threads[ds.id] ?? [];
  const busy = turns.some((t) => t.status === 'loading');
  const suggestions = useMemo(() => suggestFor(ds), [ds]);

  useEffect(() => {
    lastRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [turns.length, turns[turns.length - 1]?.status]);

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    const id = nextId.current++;
    const dsId = ds.id;
    const update = (patch: Partial<Turn>) =>
      setThreads((t) => ({ ...t, [dsId]: (t[dsId] ?? []).map((x) => (x.id === id ? { ...x, ...patch } : x)) }));
    setThreads((t) => ({ ...t, [dsId]: [...(t[dsId] ?? []), { id, question, status: 'loading' }] }));
    setInput('');
    try {
      // keep the agent timeline visible briefly even when the server answers instantly
      const [result] = await Promise.all([api.ask(dsId, question), new Promise((r) => window.setTimeout(r, 2200))]);
      update({ status: 'done', result });
    } catch (e) {
      update({ status: 'error', error: (e as Error).message });
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void ask(input);
  };

  return (
    <div className="ask">
      <div className="ask-thread">
        {turns.length === 0 && (
          <div className="ask-empty">
            <div className="ask-hero-icon">
              <Sparkles size={22} aria-hidden="true" />
            </div>
            <h2>Ask anything about {ds.name}</h2>
            <p className="muted">
              Four AI agents collaborate on every question: the <strong>Planner</strong> reads the schema, the <strong>SQL Engineer</strong> writes and
              runs DuckDB SQL, the <strong>Analyst</strong> picks the right chart, and the <strong>Narrator</strong> writes an executive summary.
            </p>
            <div className="agent-strip" aria-hidden="true">
              {AGENTS.map((a, i) => (
                <span key={a.key} className="agent-pill">
                  <a.icon size={14} /> {a.name}
                  {i < AGENTS.length - 1 && <span className="agent-arrow">→</span>}
                </span>
              ))}
            </div>
            <div className="chips center" aria-label="Suggested questions">
              {suggestions.map((s) => (
                <button key={s} className="q-chip" onClick={() => void ask(s)}>
                  {s}
                </button>
              ))}
            </div>
            {health && !health.llm && (
              <p className="muted small">Demo mode: running deterministic rule-based agents (no LLM key on the server). Phrase questions like the examples above.</p>
            )}
          </div>
        )}

        {turns.map((t, i) => (
          <article key={t.id} ref={i === turns.length - 1 ? lastRef : undefined} className="turn" aria-busy={t.status === 'loading'}>
            <div className="turn-q">
              <span className="turn-avatar" aria-hidden="true">
                You
              </span>
              <p>{t.question}</p>
            </div>
            <div className="turn-a">
              {t.status === 'loading' && <LiveTimeline />}
              {t.status === 'error' && (
                <div className="stack-sm">
                  <ErrorNote message={t.error ?? 'Something went wrong'} onRetry={() => void ask(t.question)} />
                </div>
              )}
              {t.status === 'done' && t.result && <ResultView r={t.result} onAsk={(q) => void ask(q)} datasetId={ds.id} />}
            </div>
          </article>
        ))}
      </div>

      <form className="ask-input" onSubmit={submit}>
        {turns.length > 0 && (
          <button
            type="button"
            className="icon-btn"
            onClick={() => setThreads((t) => ({ ...t, [ds.id]: [] }))}
            aria-label="Clear conversation"
            title="New conversation"
            disabled={busy}
          >
            <RotateCcw size={16} />
          </button>
        )}
        <label htmlFor="ask-q" className="sr-only">
          Ask a question about {ds.name}
        </label>
        <textarea
          id="ask-q"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void ask(input);
            }
          }}
          placeholder={`Ask a question about ${ds.name}…`}
        />
        <button type="submit" className="btn btn-primary send" disabled={busy || !input.trim()} aria-label="Send question">
          <ArrowUp size={18} />
        </button>
      </form>
    </div>
  );
}

export default function Ask() {
  return <DatasetGate>{(ds) => <Chat ds={ds} />}</DatasetGate>;
}

