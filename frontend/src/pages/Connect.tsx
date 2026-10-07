import { useCallback, useMemo, useRef, useState, type DragEvent, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2, FileUp, Lock, UploadCloud, X } from 'lucide-react';
import { api, type Connector, type ConnectorCategory, type FieldOption } from '../api';
import { useApp, useAsync } from '../context';
import { ConnectorIcon, ErrorNote, Modal, Skeleton, Spinner } from '../components/ui';

const CATEGORY_LABEL: Record<ConnectorCategory, { title: string; sub: string }> = {
  file: { title: 'Files', sub: 'Upload spreadsheets and data files' },
  cloud: { title: 'Cloud & web', sub: 'Public links, Google Sheets and hosted files' },
  database: { title: 'Databases', sub: 'Run a query against your warehouse or app DB' },
  api: { title: 'APIs & samples', sub: 'REST endpoints and built-in demo datasets' },
};
const CATEGORY_ORDER: ConnectorCategory[] = ['file', 'cloud', 'database', 'api'];

const ACCEPT: Record<string, string> = {
  csv: '.csv,.tsv,.txt',
  excel: '.xlsx,.xls',
  json: '.json,.ndjson',
  parquet: '.parquet',
  sqlite: '.db,.sqlite,.sqlite3',
};

function connectorForFile(name: string): string | null {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  if (['csv', 'tsv', 'txt'].includes(ext)) return 'csv';
  if (['xlsx', 'xls'].includes(ext)) return 'excel';
  if (['json', 'ndjson'].includes(ext)) return 'json';
  if (ext === 'parquet') return 'parquet';
  if (['db', 'sqlite', 'sqlite3'].includes(ext)) return 'sqlite';
  return null;
}

function optValue(o: FieldOption) {
  return typeof o === 'string' ? o : o.value;
}
function optLabel(o: FieldOption) {
  return typeof o === 'string' ? o : o.label ?? o.value;
}

function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function Dropzone({ file, onFile, accept, id }: { file: File | null; onFile: (f: File | null) => void; accept?: string; id: string }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };
  if (file) {
    return (
      <div className="file-chip">
        <FileUp size={18} aria-hidden="true" />
        <div className="file-chip-main">
          <div className="file-chip-name">{file.name}</div>
          <div className="muted small">{fmtBytes(file.size)}</div>
        </div>
        <button type="button" className="icon-btn" onClick={() => onFile(null)} aria-label="Remove file">
          <X size={16} />
        </button>
      </div>
    );
  }
  return (
    <div
      className={`dropzone ${over ? 'over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), input.current?.click())}
      role="button"
      tabIndex={0}
      aria-labelledby={`${id}-label`}
    >
      <UploadCloud size={28} aria-hidden="true" />
      <div id={`${id}-label`}>
        <strong>Drop a file here</strong> or click to browse
      </div>
      {accept && <div className="muted small">Accepted: {accept.split(',').join(' ')}</div>}
      <input ref={input} id={id} type="file" accept={accept} hidden onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
    </div>
  );
}

function ConnectModal({ connector, initialFile, onClose }: { connector: Connector; initialFile: File | null; onClose: () => void }) {
  const { toast, refreshDatasets, setSelectedId } = useApp();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(initialFile);
  const [name, setName] = useState('');
  const [values, setValues] = useState<Record<string, string>>(() => {
    const v: Record<string, string> = {};
    for (const f of connector.fields) if (f.type === 'select' && f.options?.length) v[f.key] = optValue(f.options[0]);
    return v;
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (connector.upload && !file) {
      setError('Please choose a file to upload.');
      return;
    }
    const missing = connector.fields.filter((f) => f.required && !(values[f.key] ?? '').trim());
    if (missing.length) {
      setError(`Please fill in: ${missing.map((m) => m.label).join(', ')}`);
      return;
    }
    setBusy(true);
    try {
      const ds = await api.connect(connector.type, values, file, name);
      await refreshDatasets();
      setSelectedId(ds.id);
      toast(`Connected “${ds.name}” · ${ds.rows.toLocaleString()} rows, ${ds.columns.length} columns`, 'success');
      onClose();
      navigate('/');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const formId = `form-${connector.type}`;
  return (
    <Modal
      open
      onClose={busy ? () => undefined : onClose}
      title={
        <span className="row gap-sm">
          <span className="conn-icon sm">
            <ConnectorIcon name={connector.icon} size={16} />
          </span>
          Connect {connector.name}
        </span>
      }
      footer={
        <>
          <button type="button" className="btn btn-ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" form={formId} className="btn btn-primary" disabled={busy}>
            {busy ? <Spinner size={16} label="Connecting…" /> : 'Connect & analyze'}
          </button>
        </>
      }
    >
      <form id={formId} className="form" onSubmit={submit} noValidate>
        <p className="muted">{connector.description}</p>
        {connector.upload && (
          <div className="field">
            <span className="label">File</span>
            <Dropzone id={`${formId}-file`} file={file} onFile={setFile} accept={ACCEPT[connector.type]} />
          </div>
        )}
        {connector.type !== 'sample' && (
          <div className="field">
            <label htmlFor={`${formId}-name`}>
              Dataset name <span className="muted">(optional)</span>
            </label>
            <input id={`${formId}-name`} value={name} onChange={(e) => setName(e.target.value)} placeholder={file ? file.name.replace(/\.[^.]+$/, '') : 'e.g. Q3 pipeline'} />
          </div>
        )}
        {connector.fields.map((f) => {
          const id = `${formId}-${f.key}`;
          const common = {
            id,
            name: f.key,
            value: values[f.key] ?? '',
            required: f.required,
            placeholder: f.placeholder,
            'aria-required': f.required,
          };
          const set = (v: string) => setValues((s) => ({ ...s, [f.key]: v }));
          return (
            <div className="field" key={f.key}>
              <label htmlFor={id}>
                {f.label}
                {f.required ? <span className="req" aria-hidden="true"> *</span> : <span className="muted"> (optional)</span>}
              </label>
              {f.type === 'textarea' ? (
                <textarea {...common} rows={4} className="mono" onChange={(e) => set(e.target.value)} />
              ) : f.type === 'select' ? (
                <select {...common} onChange={(e) => set(e.target.value)}>
                  {(f.options ?? []).map((o) => (
                    <option key={optValue(o)} value={optValue(o)}>
                      {optLabel(o)}
                    </option>
                  ))}
                </select>
              ) : (
                <input {...common} type={f.type === 'password' ? 'password' : 'text'} autoComplete={f.type === 'password' ? 'off' : undefined} onChange={(e) => set(e.target.value)} />
              )}
            </div>
          );
        })}
        {connector.category === 'database' && (
          <p className="hint row gap-sm">
            <Lock size={14} aria-hidden="true" /> Credentials are sent once to the analysis server to run the query and are not stored. Use a read-only user.
          </p>
        )}
        {error && <ErrorNote message={error} />}
      </form>
    </Modal>
  );
}

export default function Connect() {
  const { status, datasets } = useApp();
  const { data, error, loading, reload } = useAsync(status === 'ok' ? () => api.connectors() : null, [status]);
  const [active, setActive] = useState<{ connector: Connector; file: File | null } | null>(null);
  const [pageOver, setPageOver] = useState(false);
  const { toast } = useApp();

  const grouped = useMemo(() => {
    const g = new Map<ConnectorCategory, Connector[]>();
    for (const c of data ?? []) {
      const cat = (CATEGORY_ORDER.includes(c.category) ? c.category : 'api') as ConnectorCategory;
      g.set(cat, [...(g.get(cat) ?? []), c]);
    }
    return CATEGORY_ORDER.filter((c) => g.has(c)).map((c) => ({ cat: c, items: g.get(c)! }));
  }, [data]);

  const onQuickFile = useCallback(
    (f: File | null) => {
      if (!f) return;
      const type = connectorForFile(f.name);
      const conn = type ? data?.find((c) => c.type === type) : undefined;
      if (!conn) {
        toast(`Unsupported file type: ${f.name}`, 'error');
        return;
      }
      setActive({ connector: conn, file: f });
    },
    [data, toast],
  );

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <div className="eyebrow">Data sources</div>
          <h2 className="page-title">Connect your data</h2>
          <p className="muted">Bring data from files, cloud links, databases or APIs. DataRat profiles it instantly and makes it queryable by the AI agents.</p>
        </div>
        {datasets.length > 0 && (
          <div className="muted small row gap-sm">
            <CheckCircle2 size={14} className="text-good" aria-hidden="true" /> {datasets.length} dataset{datasets.length === 1 ? '' : 's'} loaded
          </div>
        )}
      </div>

      <div
        className={`quick-drop ${pageOver ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setPageOver(true);
        }}
        onDragLeave={() => setPageOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setPageOver(false);
          onQuickFile(e.dataTransfer.files?.[0] ?? null);
        }}
      >
        <UploadCloud size={24} aria-hidden="true" />
        <div>
          <strong>Quick upload</strong>
          <div className="muted small">Drop a CSV, Excel, JSON, Parquet or SQLite file — the right connector is picked automatically.</div>
        </div>
        <label className="btn btn-sm">
          Browse…
          <input
            type="file"
            hidden
            accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.ndjson,.parquet,.db,.sqlite,.sqlite3"
            onChange={(e) => {
              onQuickFile(e.target.files?.[0] ?? null);
              e.target.value = '';
            }}
            disabled={!data}
          />
        </label>
      </div>

      {error && <ErrorNote message={error} onRetry={reload} />}
      {(loading || status !== 'ok') && !data && (
        <div className="grid-conn">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="card card-body">
              <Skeleton h={36} w={36} r={10} />
              <div style={{ height: 12 }} />
              <Skeleton h={14} w="50%" />
              <div style={{ height: 8 }} />
              <Skeleton h={12} />
            </div>
          ))}
        </div>
      )}

      {grouped.map(({ cat, items }) => (
        <section key={cat} className="stack-sm" aria-labelledby={`cat-${cat}`}>
          <div>
            <h3 id={`cat-${cat}`} className="section-title">
              {CATEGORY_LABEL[cat].title}
            </h3>
            <p className="muted small">{CATEGORY_LABEL[cat].sub}</p>
          </div>
          <div className="grid-conn">
            {items.map((c) => (
              <button key={c.type} className="card conn-card" onClick={() => setActive({ connector: c, file: null })}>
                <span className={`conn-icon cat-${cat}`}>
                  <ConnectorIcon name={c.icon} />
                </span>
                <span className="conn-name">{c.name}</span>
                <span className="conn-desc">{c.description}</span>
              </button>
            ))}
          </div>
        </section>
      ))}

      {active && <ConnectModal key={active.connector.type} connector={active.connector} initialFile={active.file} onClose={() => setActive(null)} />}
    </div>
  );
}
