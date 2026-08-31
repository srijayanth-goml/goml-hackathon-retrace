import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { isHeavyDepsMissing } from "../../api/client";

interface HeavyDepsContextValue {
  /** Non-null once ANY request has 503'd with HeavyDepsMissing -- the exact
   * `detail` string app/backend/adapters.py's HeavyDepsMissing raised, so the
   * banner shows the backend's own explanation rather than a generic one. */
  detail: string | null;
  /** Call from a catch block: reportIfHeavyDepsMissing(err) is a no-op for any
   * other error, so components can call it unconditionally without an extra
   * isHeavyDepsMissing(err) check at every call site. */
  reportIfHeavyDepsMissing: (err: unknown) => void;
  /** Direct report path for a heavy-deps failure that didn't arrive as an
   * ApiError -- e.g. a background job that failed with a torch/transformers/peft
   * message in JobStatus.error rather than as an HTTP 503 (see JobStatusPanel). */
  report: (detail: string) => void;
  clear: () => void;
}

const HeavyDepsContext = createContext<HeavyDepsContextValue | null>(null);

/**
 * App-wide banner state for app/backend's 503 HeavyDepsMissing responses
 * (POST /chat and a real erasure-request job both return this when
 * torch/transformers/peft aren't installed on the backend machine -- expected on
 * this repo's own dev posture, not an exceptional error). Any component that talks
 * to the model calls reportIfHeavyDepsMissing() in its catch block; HeavyDepsBanner
 * (mounted once in App.tsx, above the tabs) is the only consumer that renders it.
 */
export function HeavyDepsProvider({ children }: { children: ReactNode }) {
  const [detail, setDetail] = useState<string | null>(null);

  const report = useCallback((detail: string) => setDetail(detail), []);

  const reportIfHeavyDepsMissing = useCallback(
    (err: unknown) => {
      if (isHeavyDepsMissing(err)) {
        report(err.message);
      }
    },
    [report],
  );

  const clear = useCallback(() => setDetail(null), []);

  const value = useMemo(
    () => ({ detail, reportIfHeavyDepsMissing, report, clear }),
    [detail, reportIfHeavyDepsMissing, report, clear],
  );

  return <HeavyDepsContext.Provider value={value}>{children}</HeavyDepsContext.Provider>;
}

export function useHeavyDeps(): HeavyDepsContextValue {
  const ctx = useContext(HeavyDepsContext);
  if (!ctx) {
    throw new Error("useHeavyDeps() must be used within a HeavyDepsProvider (see App.tsx)");
  }
  return ctx;
}
