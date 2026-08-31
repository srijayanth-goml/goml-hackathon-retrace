import type { RevisionSummary } from "../../api/types";

interface RevisionListProps {
  revisions: RevisionSummary[];
  loading: boolean;
  error: string | null;
  onOpenReport: (revision: number) => void;
}

function headlineAccuracyLabel(rev: RevisionSummary): string {
  const acc = rev.accuracy;
  if (acc.headline_accuracy == null) return "n/a";
  const pct = `${(acc.headline_accuracy * 100).toFixed(1)}%`;
  return acc.kind === "baseline_eval_summary" ? `${pct} (heldout sanity check)` : `${pct} (forget-set, after)`;
}

function requestLabel(rev: RevisionSummary): string {
  if (rev.revision === 0) return "-- baseline --";
  const req = rev.erasure_request as { entity?: string | null; attribute?: string | null } | null;
  if (!req) return "n/a";
  return [req.entity, req.attribute].filter(Boolean).join(" / ") || "n/a";
}

/**
 * The full manifest, normalized (manifest_view.py) but not reduced -- unlike
 * CompareChat's picker, this table shows every revision including its own "kind"-
 * discriminated headline accuracy so a judge can see revision-0's heldout sanity
 * check is a DIFFERENT measurement from revision-N's forget-set accuracy, never one
 * misleadingly unified number.
 */
export function RevisionList({ revisions, loading, error, onOpenReport }: RevisionListProps) {
  if (loading) return <p className="text-sm text-slate-500 dark:text-slate-400">Loading revisions...</p>;
  if (error) return <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>;
  if (revisions.length === 0) return <p className="text-sm text-slate-500 dark:text-slate-400">No revisions in the manifest yet.</p>;

  return (
    <div className="card overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:text-slate-400">
            <th className="py-2 pr-4">Revision</th>
            <th className="py-2 pr-4">Parent</th>
            <th className="py-2 pr-4">Method</th>
            <th className="py-2 pr-4">Erasure request</th>
            <th className="py-2 pr-4">Headline accuracy</th>
            <th className="py-2 pr-4">Verified</th>
            <th className="py-2 pr-4">Report</th>
          </tr>
        </thead>
        <tbody>
          {revisions.map((rev) => (
            <tr key={rev.revision} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
              <td className="py-2 pr-4 font-medium text-slate-900 dark:text-slate-100">
                revision-{rev.revision}
                {rev.label ? ` (${rev.label})` : ""}
              </td>
              <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">
                {rev.parent_revision != null ? `revision-${rev.parent_revision}` : "--"}
              </td>
              <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{rev.method ?? "--"}</td>
              <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{requestLabel(rev)}</td>
              <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{headlineAccuracyLabel(rev)}</td>
              <td className="py-2 pr-4">
                <span className={rev.has_verification_report ? "badge-pass" : "badge-neutral"}>
                  {rev.has_verification_report ? "verified" : "not verified"}
                </span>
              </td>
              <td className="py-2 pr-4">
                <button type="button" onClick={() => onOpenReport(rev.revision)} className="btn-secondary">
                  {rev.has_verification_report ? "Open report" : "View / generate"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
