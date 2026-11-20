import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { CalendarDays, Download, FlaskConical, Lightbulb, MessageCircle, Package, Phone, Play, ShoppingCart, Users, Warehouse } from 'lucide-react';
import { api, ApiError, type CallRow, type MetricRow, type PoLine, type PoStatus, type WholesaleReport } from '../api';
import { useApp } from '../context';
import { TrendForecastChart } from '../components/charts';
import { Badge, Card, EmptyState, ErrorNote, Skeleton, Spinner } from '../components/ui';
import { fmtDate, isNum } from '../format';

// Indian digit grouping: ₹12,34,567
const inr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
const fmtInr = (v: number | null | undefined) => (isNum(v) ? inr.format(v) : '—');
const fmtQty = (v: number | null | undefined) => (isNum(v) ? Math.round(v).toLocaleString('en-IN') : '—');

const STATUS_TONE: Record<PoStatus, 'bad' | 'warn' | 'accent' | 'neutral' | 'good'> = {
  URGENT: 'bad',
  EXPEDITE: 'warn',
  ORDER: 'accent',
  'CHECK STOCK': 'neutral',
  OK: 'good',
};

const ERP_OPTIONS = [
  { value: 'auto', label: 'Detect automatically' },
  { value: 'tally', label: 'Tally Prime (sales register / XML)' },
  { value: 'marg', label: 'Marg ERP' },
  { value: 'vyapar', label: 'Vyapar' },
  { value: 'canonical', label: 'Canonical columns' },
];

function Stat({ label, value, sub, icon }: { label: string; value: string; sub?: string; icon: ReactNode }) {
  return (
    <div className="card kpi">
      <div className="kpi-label row gap-sm">
        {icon}
        {label}
      </div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-meta muted small">{sub}</div>}
    </div>
  );
}

function PoTable({ rows }: { rows: PoLine[] }) {
  return (
    <div className="table-wrap" style={{ maxHeight: 460 }} tabIndex={0} role="region" aria-label="Purchase-order plan">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Status</th>
            <th scope="col">Item</th>
            <th scope="col" className="num">Order</th>
            <th scope="col" className="num">On hand</th>
            <th scope="col" className="num">On order</th>
            <th scope="col" className="num">Reorder point</th>
            <th scope="col" className="num">Order-up-to</th>
            <th scope="col" className="num">Runs out in</th>
            <th scope="col" className="num">Next 7 / 15 / 30 days</th>
            <th scope="col" className="num">PO value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.sku}>
              <td>
                <Badge tone={STATUS_TONE[r.status] ?? 'neutral'}>{r.status}</Badge>
              </td>
              <td>
                <div>{r.item_name}</div>
                <div className="muted small">
                  {r.supplier} · lead time {r.lead_time_days} d{r.forecast_method === 'cold_start' ? ' · new item' : ''}
                </div>
              </td>
              <td className="num">
                {r.order_packs > 0 ? (
                  <strong>
                    {r.order_packs} {r.pack_unit}
                  </strong>
                ) : (
                  '—'
                )}
                {r.order_packs > 0 && <div className="muted small">{fmtQty(r.order_qty)} {r.unit}</div>}
              </td>
              <td className="num">{fmtQty(r.on_hand)}</td>
              <td className="num">{fmtQty(r.on_order)}</td>
              <td className="num">{fmtQty(r.reorder_point)}</td>
              <td className="num">{fmtQty(r.order_up_to)}</td>
              <td className="num">{isNum(r.stockout_in_days) ? `${r.stockout_in_days} d` : '—'}</td>
              <td className="num">
                {fmtQty(r.forecast_7d)} / {fmtQty(r.forecast_15d)} / {fmtQty(r.forecast_30d)} {r.unit}
              </td>
              <td className="num">{r.est_po_value ? fmtInr(r.est_po_value) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CallTable({ rows }: { rows: CallRow[] }) {
  return (
    <div className="table-wrap" style={{ maxHeight: 520 }} tabIndex={0} role="region" aria-label="Buyer outreach list">
      <table className="data-table ws-calls">
        <thead>
          <tr>
            <th scope="col" className="num">#</th>
            <th scope="col">Buyer</th>
            <th scope="col">Action</th>
            <th scope="col">Suggested items</th>
            <th scope="col" className="num">Expected value</th>
            <th scope="col" className="num">p(order)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.buyer_id}>
              <td className="num">{r.rank}</td>
              <td>
                <div>{r.buyer_name}</div>
                <div className="muted small">
                  {r.segment} · {r.phone}
                </div>
              </td>
              <td>
                <div className="row gap-sm">
                  {r.channel === 'WhatsApp' ? <MessageCircle size={14} aria-hidden="true" /> : <Phone size={14} aria-hidden="true" />}
                  <span>{r.channel}</span>
                </div>
                <Badge tone={r.action === 'Win-back call' ? 'warn' : 'accent'}>{r.action}</Badge>
              </td>
              <td className="ws-wrap">
                <div>{r.items}</div>
                <div className="muted small">{r.why}</div>
                {r.check_stock && <div className="small ws-check">Check stock: {r.check_stock}</div>}
              </td>
              <td className="num">{fmtInr(r.expected_value)}</td>
              <td className="num">{r.max_probability.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricsTable({ rows }: { rows: MetricRow[] }) {
  if (!rows.length) return <p className="muted">No backtest available.</p>;
  const cols = Object.keys(rows[0]).filter((c) => c !== 'model');
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label="Model comparison">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Model</th>
            {cols.map((c) => (
              <th key={c} scope="col" className="num">
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model}>
              <td>{i === 0 ? <strong>{r.model}</strong> : r.model}</td>
              {cols.map((c) => {
                const v = r[c];
                return (
                  <td key={c} className={`num ${v === null ? 'null' : ''}`}>
                    {isNum(v) ? v.toFixed(3) : '—'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunForm({ busy, onRun }: { busy: boolean; onRun: (body: Parameters<typeof api.wholesaleRun>[0]) => void }) {
  const { datasets } = useApp();
  const [inv, setInv] = useState('');
  const [stock, setStock] = useState('');
  const [items, setItems] = useState('');
  const [parties, setParties] = useState('');
  const [erp, setErp] = useState('auto');
  const opts = datasets.filter((d) => !d.id.startsWith('wholesale_'));
  const pick = (id: string, label: string, value: string, set: (v: string) => void, optional = true) => (
    <div className="field">
      <label htmlFor={id}>
        {label} {optional && <span className="muted">(optional)</span>}
      </label>
      <select id={id} value={value} onChange={(e) => set(e.target.value)}>
        <option value="">{optional ? '— none —' : 'Choose a dataset'}</option>
        {opts.map((d) => (
          <option key={d.id} value={d.id}>
            {d.name} ({d.rows.toLocaleString()} rows)
          </option>
        ))}
      </select>
    </div>
  );
  return (
    <form
      className="ws-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (!inv) return;
        onRun({
          invoices_dataset_id: inv,
          stock_dataset_id: stock || undefined,
          items_dataset_id: items || undefined,
          parties_dataset_id: parties || undefined,
          erp,
        });
      }}
    >
      {pick('ws-inv', 'Sales register / invoice export', inv, setInv, false)}
      <div className="field">
        <label htmlFor="ws-erp">ERP format</label>
        <select id="ws-erp" value={erp} onChange={(e) => setErp(e.target.value)}>
          {ERP_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      {pick('ws-stock', 'Daily closing stock', stock, setStock)}
      {pick('ws-items', 'Item master (pack size, lead time, MOQ)', items, setItems)}
      {pick('ws-parties', 'Party master (segment, phone, WhatsApp opt-in)', parties, setParties)}
      <div className="ws-form-actions">
        <button className="btn btn-primary" type="submit" disabled={busy || !inv}>
          {busy ? <Spinner size={14} /> : <Play size={16} aria-hidden="true" />} Train and plan
        </button>
        <span className="hint">Upload exports on the Connect page first (CSV or Excel). Training takes a few seconds to a minute.</span>
      </div>
    </form>
  );
}

function Report({ r }: { r: WholesaleReport }) {
  // default to the most urgent item in the PO plan (the plan is sorted by status, then value)
  const firstSku = r.po_plan[0]?.sku ?? r.forecast[0]?.sku ?? '';
  const [sku, setSku] = useState(firstSku);
  useEffect(() => setSku(firstSku), [r, firstSku]);
  const item = useMemo(() => r.forecast.find((f) => f.sku === sku), [r, sku]);
  const s = r.summary;
  const dm = r.metrics.demand;
  const rm = r.metrics.reorder;
  return (
    <div className="stack">
      <div className="grid-kpi">
        <Stat
          label="Items to order"
          value={`${s.items_to_order} of ${s.items}`}
          sub={`${s.urgent_items} urgent · ${s.expedite_items ?? 0} to expedite`}
          icon={<ShoppingCart size={14} aria-hidden="true" />}
        />
        <Stat label="Purchase orders" value={fmtInr(s.po_value)} sub="at estimated cost" icon={<Package size={14} aria-hidden="true" />} />
        <Stat
          label="Buyers to contact"
          value={String(s.buyers_to_contact)}
          sub={`${s.calls} calls · ${s.whatsapp} WhatsApp`}
          icon={<Users size={14} aria-hidden="true" />}
        />
        <Stat label="Expected orders" value={fmtInr(s.expected_order_value)} sub="from today's list" icon={<Phone size={14} aria-hidden="true" />} />
      </div>

      <div className="grid-2-1">
        <Card
          title="Demand forecast"
          subtitle={item ? `${item.item_name}: next 7 / 15 / 30 days ${fmtQty(item.h7)} / ${fmtQty(item.h15)} / ${fmtQty(item.h30)} ${item.unit}${item.method === 'cold_start' ? ' (new item, prior-based)' : ''}` : undefined}
          actions={
            <select aria-label="Item" value={sku} onChange={(e) => setSku(e.target.value)} className="ws-item-select">
              {r.forecast.map((f) => (
                <option key={f.sku} value={f.sku}>
                  {f.item_name}
                </option>
              ))}
            </select>
          }
        >
          {item && item.history.length > 1 ? (
            <TrendForecastChart key={item.sku} history={item.history} forecast={item.daily} metric={`demand_${item.unit}`} />
          ) : (
            <p className="muted">Not enough history to chart.</p>
          )}
          <p className="muted small">Shaded band: about 80% of backtest errors. History is demand with stockout days corrected.</p>
        </Card>
        <Card title="Today's highlights" subtitle={`Plan for ${fmtDate(s.as_of, 'long')}`}>
          <ul className="highlights">
            {r.highlights.map((h, i) => (
              <li key={i}>
                <Lightbulb size={16} aria-hidden="true" />
                <span>{h}</span>
              </li>
            ))}
          </ul>
          {s.upcoming_events.length > 0 && (
            <>
              <div className="section-title" style={{ marginTop: 12 }}>
                Coming up
              </div>
              <ul className="highlights">
                {s.upcoming_events.map((e) => (
                  <li key={e.date + e.festival}>
                    <CalendarDays size={16} aria-hidden="true" />
                    <span>
                      {e.festival} in {e.in_days} d · {e.commodities.join(', ')}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Card>
      </div>

      <Card title="Purchase-order plan" subtitle="Min/max policy: order up to the max when stock on hand + on order falls to the reorder point" pad={false}>
        <PoTable rows={r.po_plan} />
      </Card>

      <Card title="Buyer outreach list" subtitle={`Ranked by probability × order value · top ${r.call_list.length}`} pad={false}>
        {r.call_list.length ? <CallTable rows={r.call_list} /> : <p className="card-body muted">No buyers are due today.</p>}
      </Card>

      <div className="stack">
        <Card title="Demand model backtest" subtitle={dm ? `${dm.split_date} to ${dm.end_date}, ${dm.test_origins} weekly origins, ${dm.evaluated_on}` : undefined}>
          <MetricsTable rows={dm?.models ?? []} />
        </Card>
        <Card title="Buyer reorder backtest" subtitle={rm ? `${rm.test_from} to ${rm.test_to}, base rate ${(rm.base_rate * 100).toFixed(1)}%, list size ${rm.k}` : undefined}>
          <MetricsTable rows={rm?.models ?? []} />
        </Card>
      </div>
      <p className="muted small">
        Model backend: {r.model.backend ?? '—'} · trained through {r.model.trained_through ?? '—'} · {s.history.lines.toLocaleString()} invoice lines,{' '}
        {s.history.buyers} buyers ({s.history.from} to {s.history.to}). Tables are also in SQL Lab: {(r.datasets ?? []).join(', ') || '—'}.
      </p>
    </div>
  );
}

export default function Wholesale() {
  const { status, refreshDatasets, toast } = useApp();
  const [report, setReport] = useState<WholesaleReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  useEffect(() => {
    if (status !== 'ok') return;
    let cancelled = false;
    setLoading(true);
    api
      .wholesaleReport()
      .then((r) => !cancelled && setReport(r))
      .catch((e: ApiError) => !cancelled && e.status !== 404 && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [status]);

  const run = async (fn: () => Promise<WholesaleReport>, what: string) => {
    setBusy(true);
    setError(null);
    try {
      const r = await fn();
      setReport(r);
      setShowForm(false);
      void refreshDatasets();
      toast(`${what}: ${r.summary.items_to_order} item(s) to order, ${r.summary.buyers_to_contact} buyers to contact`, 'success');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <div className="eyebrow">Wholesale planner</div>
          <h2 className="page-title">Purchase orders and buyer outreach</h2>
          <p className="muted small" style={{ maxWidth: 720 }}>
            Forecasts 7, 15 and 30-day demand per item from your billing exports, turns it into purchase-order quantities with safety stock, and
            ranks which buyers are about to reorder or have gone quiet. Built for staples traders using Tally Prime, Marg or Vyapar.
          </p>
        </div>
        <div className="row gap-sm wrap">
          <button className="btn" onClick={() => setShowForm((v) => !v)} disabled={busy}>
            <Warehouse size={16} aria-hidden="true" /> Use my data
          </button>
          <button className="btn btn-primary" onClick={() => void run(() => api.wholesaleDemo(), 'Demo plan ready')} disabled={busy || status !== 'ok'}>
            {busy ? <Spinner size={14} /> : <FlaskConical size={16} aria-hidden="true" />} Run on synthetic data
          </button>
          {report && (
            <a className="btn" href={api.wholesaleXlsxUrl()} download>
              <Download size={16} aria-hidden="true" /> Excel
            </a>
          )}
        </div>
      </div>

      {showForm && (
        <Card title="Plan from your exports" subtitle="Pick datasets you connected; column names are matched automatically">
          <RunForm busy={busy} onRun={(body) => void run(() => api.wholesaleRun(body), 'Plan ready')} />
        </Card>
      )}

      {error && <ErrorNote message={error} />}
      {busy && <Spinner label="Training the demand and reorder models…" />}

      {loading && status === 'ok' ? (
        <div className="card card-body">
          <Skeleton h={240} />
        </div>
      ) : report ? (
        <Report r={report} />
      ) : (
        !busy && (
          <EmptyState icon={<Warehouse size={28} />} title="No plan yet">
            <p>Run the planner on synthetic invoices (rice, sugar, pulses and oil sold to kirana stores, hotels and caterers), or on your own exports.</p>
            <p className="muted small">Synthetic data only; nothing is sent to WhatsApp from here.</p>
          </EmptyState>
        )
      )}
    </div>
  );
}
