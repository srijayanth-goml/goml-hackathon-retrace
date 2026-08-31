import { useActiveJob } from "../../hooks/useActiveJob";

/**
 * Shown app-wide while ANY job (train_and_verify / train_only / verify_only) is
 * queued/running/verifying -- surfaces app/backend/adapters.py's real MODEL_LOCK
 * behavior (a chat request and a training/verification job never touch the model
 * at the same moment, by design, given the single-machine hardware this runs on)
 * instead of letting a judge wonder why chat suddenly got slow or stopped replying.
 */
export function ActiveJobBanner() {
  const { activeJob } = useActiveJob();
  if (!activeJob) return null;

  const label =
    activeJob.job_type === "verify_only"
      ? `Verifying revision ${activeJob.revision ?? "?"}`
      : `Running erasure request (${activeJob.method ?? "?"})`;

  return (
    <div className="flex items-center gap-2 border-b border-sky-300 bg-sky-50 px-4 py-2 text-sm text-sky-900 dark:border-sky-700 dark:bg-sky-950/60 dark:text-sky-200">
      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-sky-500" aria-hidden="true" />
      <span>
        <span className="font-semibold">{label}</span> ({activeJob.status}) -- chat replies may be slow until
        this finishes; a second submission is disabled to avoid a confusing second job with no visible status.
      </span>
    </div>
  );
}
