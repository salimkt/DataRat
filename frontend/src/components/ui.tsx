import { useEffect, useRef, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  AlertTriangle,
  BarChart3,
  Braces,
  Cloud,
  Database,
  FileJson,
  FileSpreadsheet,
  FileText,
  FileUp,
  FlaskConical,
  Globe,
  HardDrive,
  Layers,
  Link,
  Loader2,
  Package,
  Server,
  Sheet,
  Table,
  Webhook,
  X,
  type LucideIcon,
} from 'lucide-react';
import { fmtCell, isNum } from '../format';

// ---------- Icon by lucide name (curated to keep the bundle small) ----------

const ICONS: Record<string, LucideIcon> = {
  filespreadsheet: FileSpreadsheet,
  filetext: FileText,
  filejson: FileJson,
  fileup: FileUp,
  file: FileText,
  database: Database,
  server: Server,
  harddrive: HardDrive,
  globe: Globe,
  link: Link,
  link2: Link,
  sheet: Sheet,
  table: Table,
  table2: Table,
  cloud: Cloud,
  braces: Braces,
  webhook: Webhook,
  layers: Layers,
  package: Package,
  boxes: Package,
  barchart3: BarChart3,
  chartbar: BarChart3,
  sparkles: Layers,
  flaskconical: FlaskConical,
  flask: FlaskConical,
};

export function ConnectorIcon({ name, size = 20 }: { name: string; size?: number }) {
  const key = (name || '').toLowerCase().replace(/[^a-z0-9]/g, '');
  const Icon = ICONS[key] ?? Database;
  return <Icon size={size} aria-hidden="true" />;
}

// ---------- Basic building blocks ----------

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
  pad = true,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h3 className="card-title">{title}</h3>}
            {subtitle && <p className="card-sub">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      <div className={pad ? 'card-body' : ''}>{children}</div>
    </section>
  );
}

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return (
    <span className="spinner-wrap" role="status">
      <Loader2 size={size} className="spin" aria-hidden="true" />
      {label ? <span>{label}</span> : <span className="sr-only">Loading</span>}
    </span>
  );
}

export function Skeleton({ h = 16, w = '100%', r = 6 }: { h?: number | string; w?: number | string; r?: number }) {
  return <div className="skeleton" style={{ height: h, width: w, borderRadius: r }} aria-hidden="true" />;
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      {icon && <div className="empty-icon">{icon}</div>}
      <h4>{title}</h4>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-note" role="alert">
      <AlertTriangle size={16} aria-hidden="true" />
      <span>{message}</span>
      {onRetry && (
        <button className="btn btn-ghost btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'bad' | 'warn' | 'accent' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

// ---------- Modal ----------

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    const t = window.setTimeout(() => {
      const el = ref.current?.querySelector<HTMLElement>('input, textarea, select, button:not(.modal-close)');
      el?.focus();
    }, 30);
    return () => {
      document.removeEventListener('keydown', onKey);
      window.clearTimeout(t);
      prev?.focus?.();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? 'modal-wide' : ''}`} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : undefined} ref={ref}>
        <header className="modal-head">
          <h2>{title}</h2>
          <button className="icon-btn modal-close" onClick={onClose} aria-label="Close dialog">
            <X size={18} />
          </button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-foot">{footer}</footer>}
      </div>
    </div>
  );
}

// ---------- Markdown ----------

export function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown>{children || ''}</ReactMarkdown>
    </div>
  );
}

/** Inline markdown (bold/italic/code) without wrapping <p>. */
export function InlineMarkdown({ children }: { children: string }) {
  return (
    <span className="md-inline">
      <ReactMarkdown allowedElements={['p', 'strong', 'em', 'code', 'a']} unwrapDisallowed components={{ p: ({ children: c }) => <>{c}</> }}>
        {children || ''}
      </ReactMarkdown>
    </span>
  );
}

// ---------- Data table ----------

export function DataTable({
  columns,
  rows,
  maxRows = 200,
  maxHeight = 420,
  caption,
}: {
  columns: string[];
  rows: unknown[][];
  maxRows?: number;
  maxHeight?: number | string;
  caption?: string;
}) {
  const shown = rows.slice(0, maxRows);
  const numericCols = columns.map((_, i) => shown.length > 0 && shown.every((r) => r[i] === null || r[i] === undefined || isNum(r[i])));
  return (
    <div className="table-wrap" style={{ maxHeight }} tabIndex={0} role="region" aria-label={caption ?? 'Data table'}>
      <table className="data-table">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c, i) => (
              <th key={c + i} className={numericCols[i] ? 'num' : ''} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r, ri) => (
            <tr key={ri}>
              {columns.map((_, ci) => {
                const v = r[ci];
                return (
                  <td key={ci} className={`${numericCols[ci] ? 'num' : ''} ${v === null || v === undefined ? 'null' : ''}`}>
                    {fmtCell(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > maxRows && (
        <div className="table-more">
          Showing {maxRows.toLocaleString()} of {rows.length.toLocaleString()} rows
        </div>
      )}
    </div>
  );
}

// ---------- Segmented control ----------

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          title={o.title}
          className={value === o.value ? 'active' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
