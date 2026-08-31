import { useJobPolling } from "../../hooks/useJobPolling";
import { useHeavyDeps } from "../layout/HeavyDepsContext";
import type { JobStatus, JobStatusValue } from "../../api/types";
import { useEffect } from "react";

const STEPS: JobStatusValue[] = ["queued", "running", "verifying", "done"];

interface JobStatusPanelProps {
  jobId: string;
  onSettled?: (job: JobStatus) => void;
}

function Stepper({ status }: { status: JobStatusValue }) {
  const failed = status === "failed";
  const currentIndex = failed ? STEPS.length - 1 : STEPS.indexOf(status);

  return (
    <div className="flex items-center gap-1 text-xs">
      {STEPS.map((step, i) => {
        const reached = !failed && i <= currentIndex;
        return (
          <span key={step} className="flex items-center gap-1">
            <span
              className={
                "rounded-full px-2 py-0.5 " +
                (reached ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900" : "bg-slate-200 text-slate-500 dark:bg-slate-700 dark:text-slate-400")
              }
            >
              {step}
            </span>
            {i < STEPS.length - 1 && <span className="text-slate-300 dark:text-slate-600">&rarr;</span>}
          </span>
        );
      })}
      {failed && <span className="badge-fail ml-1">failed</span>}
    </div>
  );
}

/**
 * Renders useJobPolling()'s state: a stepper (queued -> running -> verifying ->
 * done/failed) plus a scrolling log_tail view (unlearning.train.run and
 * verification.run_verification.run's own stdout, last LOG_TAIL_MAX_LINES per
 * app/backend/config.py) -- makes the job look and feel actually live rather than a
 * canned progress bar, per this module's whole reason for existing.
 */
export function JobStatusPanel({ jobId, onSettled }: JobStatusPanelProps) {
  const { report } = useHeavyDeps();
  const { job, error } = useJobPolling(jobId, onSettled);

  useEffect(() => {
    if (job?.status === "failed" && job.error && /torch|transformers|peft/i.test(job.error)) {
      // A job that failed specifically because torch/transformers/peft aren't
      // installed is exactly HeavyDepsMissing's failure mode surfacing through the
      // background worker rather than an HTTP 503 -- route it to the same banner.
      report(job.error);
    }
  }, [job, report]);

  if (error && !job) {
    return <p className="text-sm text-rose-600 dark:text-rose-400">Could not fetch job status: {error}</p>;
  }
  if (!job) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">Loading job status...</p>;
  }

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
          Job {job.job_id.slice(0, 8)} -- {job.job_type}
        </span>
        <Stepper status={job.status} />
      </div>

      {job.status === "done" && job.revision != null && (
        <p className="text-sm text-emerald-700 dark:text-emerald-400">
          Done -- revision-{job.revision} is now chattable in Compare &amp; Chat and reportable in Reports.
        </p>
      )}
      {job.status === "failed" && (
        <p className="text-sm text-rose-600 dark:text-rose-400">Failed: {job.error ?? "no error message recorded"}</p>
      )}

      {job.log_tail.length > 0 && (
        <pre className="max-h-40 overflow-y-auto rounded-md bg-slate-950 p-2 text-xs text-slate-200">
          {job.log_tail.join("\n")}
        </pre>
      )}
    </div>
  );
}
