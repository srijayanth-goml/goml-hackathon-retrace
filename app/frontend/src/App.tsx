import { useState } from "react";
import { HeavyDepsProvider } from "./components/layout/HeavyDepsContext";
import { HeavyDepsBanner } from "./components/layout/HeavyDepsBanner";
import { ActiveJobBanner } from "./components/layout/ActiveJobBanner";
import { TabNav, type TabKey } from "./components/layout/TabNav";
import { CompareChat } from "./components/chat/CompareChat";
import { ErasureRequestForm } from "./components/erasure/ErasureRequestForm";
import { RevisionList } from "./components/revisions/RevisionList";
import { ReportView } from "./components/report/ReportView";
import { useRevisions } from "./hooks/useRevisions";

function AppShell() {
  const [tab, setTab] = useState<TabKey>("compare");
  const [reportRevision, setReportRevision] = useState(0);
  const { revisions, loading, error, refresh } = useRevisions();

  const openReport = (revision: number) => {
    setReportRevision(revision);
    setTab("reports");
  };

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-950">
      <header className="border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
        <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100">ReTrace -- Erasure Console</h1>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Module 6 (App UI) against Module 5's live API -- chat with any revision, submit an erasure request, and
          read the generated Erasure Report.
        </p>
      </header>

      <HeavyDepsBanner />
      <ActiveJobBanner />
      <TabNav active={tab} onChange={setTab} />

      <main className="mx-auto max-w-5xl p-4">
        {tab === "compare" && <CompareChat revisions={revisions} />}
        {tab === "erasure" && <ErasureRequestForm onJobDone={() => refresh()} />}
        {tab === "revisions" && (
          <RevisionList revisions={revisions} loading={loading} error={error} onOpenReport={openReport} />
        )}
        {tab === "reports" && (
          <ReportView revisions={revisions} revision={reportRevision} onRevisionChange={setReportRevision} />
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <HeavyDepsProvider>
      <AppShell />
    </HeavyDepsProvider>
  );
}
