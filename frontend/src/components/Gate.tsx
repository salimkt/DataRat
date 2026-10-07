import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Database, Plug } from 'lucide-react';
import type { Dataset } from '../api';
import { useApp } from '../context';
import { EmptyState, ErrorNote, Skeleton } from './ui';

/** Renders children only when the server is up and a dataset is selected. */
export function DatasetGate({ children }: { children: (ds: Dataset) => ReactNode }) {
  const { status, selected, datasets, datasetsLoading, datasetsError, refreshDatasets } = useApp();
  if (status !== 'ok' || (datasetsLoading && datasets.length === 0)) {
    return (
      <div className="stack" aria-busy="true">
        <div className="grid-kpi">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="card card-body">
              <Skeleton h={12} w="40%" />
              <div style={{ height: 12 }} />
              <Skeleton h={28} w="60%" />
              <div style={{ height: 12 }} />
              <Skeleton h={32} />
            </div>
          ))}
        </div>
        <div className="card card-body">
          <Skeleton h={260} />
        </div>
      </div>
    );
  }
  if (datasetsError) return <ErrorNote message={datasetsError} onRetry={() => void refreshDatasets()} />;
  if (!selected) {
    return (
      <EmptyState icon={<Database size={28} />} title="No dataset selected">
        <p>Connect a data source or reload a sample dataset to get started.</p>
        <Link to="/connect" className="btn btn-primary">
          <Plug size={16} aria-hidden="true" /> Connect data
        </Link>
      </EmptyState>
    );
  }
  return <>{children(selected)}</>;
}
