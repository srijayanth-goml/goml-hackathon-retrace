import { useEffect, useState } from "react";
import { ApiError, getReport, postGenerateReport } from "../../api/client";
import { useHeavyDeps } from "../layout/HeavyDepsContext";
import { JobStatusPanel } from "../erasure/JobStatusPanel";
import { RevisionPicker } from "../chat/RevisionPicker";
import { WhatWasTargeted } from "./WhatWasTargeted";
import { WhatWasDone } from "./WhatWasDone";
import { VerificationResults } from "./VerificationResults";
import { ImpactAssessment } from "./ImpactAssessment";
import { KeyTakeaways } from "./KeyTakeaways";
import { RawReportJson } from "./RawReportJson";
import type { ErasureReport, RevisionSummary } from "../../api/types";

interface ReportViewProps {
  revisions: RevisionSummary[];
  revision: number;
  onRevisionChange: (revision: number) => void;
}

/**
 * Fetches GET /reports/{revision}. On the backend's explicit "not verified yet"
 * 404 (routes/reports.py) or the baseline's "nothing to verify it against" 400,
 * shows that message verbatim -- never a bare 404 page -- plus a working
 * "Generate Report" button routed through the same job queue as an erasure
 * request, per plan.md's Module 6 design.
 */
export function ReportView({ revisions, revision, onRevisionChange }: ReportViewProps) {
  const { reportIfHeavyDepsMissing } = useHeavyDeps();
  const [report, setReport] = useState<ErasureReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [genJobId, setGenJobId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setReport(null);
    setNotice(null);
    setGenJobId(null);

    getReport(revision)
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        reportIfHeavyDepsMissing(err);
        if (err instanceof ApiError) {
          setNotice(err.message);
        } else {
          setNotice(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [revision, reportIfHeavyDepsMissing]);

  const generate = () => {
    setGenerating(true);
    postGenerateReport(revision)
      .then((job) => setGenJobId(job.job_id))
      .catch((err: unknown) => {
        reportIfHeavyDepsMissing(err);
        setNotice(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setGenerating(false));
  };

  const canGenerate = revision !== 0; // revision-0 is the baseline -- routes/reports.py rejects it outright

  return (
    <div className="space-y-4">
      <div className="card flex items-center justify-between">
        <span className="text-sm font-medium text-slate-600 dark:text-slate-300">Viewing report for:</span>
        <RevisionPicker revisions={revisions} value={revision} onChange={onRevisionChange} />
      </div>

      {loading && <p className="text-sm text-slate-500 dark:text-slate-400">Loading report...</p>}

      {!loading && notice && !report && (
        <div className="card space-y-2">
          <p className="text-sm text-slate-700 dark:text-slate-200">{notice}</p>
          {canGenerate && (
            <button type="button" onClick={generate} disabled={generating || !!genJobId} className="btn">
              {generating ? "Requesting..." : "Generate Report"}
            </button>
          )}
        </div>
      )}

      {genJobId && (
        <JobStatusPanel
          jobId={genJobId}
          onSettled={(job) => {
            if (job.status === "done") {
              // Re-fetch now that verification.run_verification.run() has written
              // the report to disk -- same revision, now with a report to show.
              getReport(revision).then(setReport).catch(() => undefined);
            }
          }}
        />
      )}

      {!loading && report && (
        <>
          <WhatWasTargeted report={report} />
          <WhatWasDone report={report} />
          <VerificationResults vr={report.verification_results} />
          <ImpactAssessment impact={report.impact_assessment} />
          <KeyTakeaways takeaways={report.key_takeaways} />
          <RawReportJson report={report} />
        </>
      )}
    </div>
  );
}
