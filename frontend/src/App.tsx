import { Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import Overview from './pages/Overview';
import Connect from './pages/Connect';
import Datasets from './pages/Datasets';
import Ask from './pages/Ask';
import SqlLab from './pages/SqlLab';
import Brief from './pages/Brief';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/connect" element={<Connect />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/ask" element={<Ask />} />
        <Route path="/sql" element={<SqlLab />} />
        <Route path="/brief" element={<Brief />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
