import { useEffect, useState } from "react";
import { listJobs } from "../api/client";
import { ACTIVE_JOB_STATUSES, type JobStatus } from "../api/types";

const CHECK_INTERVAL_MS = 3000;

/**
 * Derives "is ANY job currently running" from GET /jobs, polled on a slower
 * interval than useJobPolling (this only backs an app-wide banner + disabling
 * submit buttons, not a live progress view). Exists because app/backend/jobs.py
 * runs exactly ONE background worker on ONE global model lock (adapters.MODEL_LOCK)
 * -- a second submission while a job is running wouldn't fail, it would just queue
 * silently behind the first with no visible status of its own, which is confusing
 * mid-demo. This hook lets ErasureRequestForm and the reports "Generate" action
 * disable themselves instead of accepting a submission the UI can't clearly track.
 */
export function useActiveJob(): { activeJob: JobStatus | null } {
  const [activeJob, setActiveJob] = useState<JobStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const check = () => {
      listJobs()
        .then((jobs) => {
          if (cancelled) return;
          const running = jobs.find((j) => (ACTIVE_JOB_STATUSES as readonly string[]).includes(j.status));
          setActiveJob(running ?? null);
        })
        .catch(() => {
          // Backend unreachable -- HeavyDepsBanner (driven by actual request
          // failures elsewhere) already covers surfacing that; this hook just
          // stays silent rather than duplicating an error banner.
        })
        .finally(() => {
          if (!cancelled) timer = setTimeout(check, CHECK_INTERVAL_MS);
        });
    };

    check();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return { activeJob };
}
