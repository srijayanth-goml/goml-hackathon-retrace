import { useCallback, useEffect, useState } from "react";
import { getRevisions } from "../api/client";
import type { RevisionSummary } from "../api/types";

interface UseRevisionsResult {
  revisions: RevisionSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/**
 * GET /revisions on mount, plus an exposed refresh() -- called again after any job
 * reaches "done" (see useJobPolling), since a completed job already means
 * adapters.refresh() ran server-side and the new revision is chattable immediately;
 * this hook is just how the UI finds out. Never polls on its own -- the manifest
 * only changes when a job finishes, and JobStatusPanel already knows when that is.
 */
export function useRevisions(): UseRevisionsResult {
  const [revisions, setRevisions] = useState<RevisionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getRevisions()
      .then((data) => {
        if (!cancelled) {
          setRevisions(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  return { revisions, loading, error, refresh };
}
