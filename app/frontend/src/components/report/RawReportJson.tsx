import type { ErasureReport } from "../../api/types";

/**
 * A collapsed safety net: the full report JSON, pretty-printed. The structured
 * sections above render every top-level field this repo's verification code
 * currently produces, but this panel stays present regardless -- so a nested field
 * Module 4 adds later is still visible to a judge who wants to dig in, without a
 * frontend release being required first. Matches verification/report.py's own
 * "never silently omit" framing of itself (see e.g. reference_comparison.py's
 * explicit `available: false` branch instead of an omitted section).
 */
export function RawReportJson({ report }: { report: ErasureReport }) {
  return (
    <details className="card">
      <summary className="cursor-pointer font-semibold text-slate-900 dark:text-slate-100">
        Raw report JSON (full fidelity)
      </summary>
      <pre className="mt-2 max-h-96 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-200">
        {JSON.stringify(report, null, 2)}
      </pre>
    </details>
  );
}
