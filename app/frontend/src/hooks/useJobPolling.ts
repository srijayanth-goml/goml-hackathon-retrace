import { useEffect, useRef, useState } from "react";
import { getJob } from "../api/client";
import { ACTIVE_JOB_STATUSES, type JobStatus } from "../api/types";

const POLL_INTERVAL_MS = 1500;

interface UseJobPollingResult {
  job: JobStatus | null;
  error: string | null;
}

/**
 * Given a job_id, polls GET /jobs/{id} at POLL_INTERVAL_MS while status is
 * queued/running/verifying, stops the instant it's done/failed. `onSettled` fires
 * exactly once, when the job first reaches a terminal state -- JobStatusPanel uses
 * it to trigger useRevisions().refresh() so a completed erasure request's new
 * revision shows up without a page reload.
 *
 * Passing `jobId: null` (no job submitted yet, or a fresh screen) is a no-op --
 * this hook doesn't own submission, only polling of a job that's already running.
 */
export function useJobPolling(jobId: string | null, onSettled?: (job: JobStatus) => void): UseJobPollingResult {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const onSettledRef = useRef(onSettled);
  onSettledRef.current = onSettled;

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let settledFired = false;

    const poll = () => {
      getJob(jobId)
        .then((data) => {
          if (cancelled) return;
          setJob(data);
          setError(null);
          const stillActive = (ACTIVE_JOB_STATUSES as readonly string[]).includes(data.status);
          if (stillActive) {
            timer = setTimeout(poll, POLL_INTERVAL_MS);
          } else if (!settledFired) {
            settledFired = true;
            onSettledRef.current?.(data);
          }
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setError(err instanceof Error ? err.message : String(err));
          // A transient network hiccup shouldn't permanently stop polling a job a
          // judge is watching -- retry on the same interval rather than giving up.
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        });
    };

    poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  return { job, error };
}
