import type { RevisionSummary } from "../../api/types";

interface RevisionPickerProps {
  revisions: RevisionSummary[];
  value: number;
  onChange: (revision: number) => void;
  /** Revisions to exclude from the list -- CompareChat uses this so the right-hand
   * picker can't also select whatever the left pane (pinned to revision 0) is on. */
  exclude?: number[];
}

function labelFor(rev: RevisionSummary): string {
  if (rev.revision === 0) return "revision-0 (baseline)";
  const req = rev.erasure_request as { entity?: string; attribute?: string } | null;
  const target = req ? [req.entity, req.attribute].filter(Boolean).join(" / ") : null;
  return `revision-${rev.revision}${target ? ` -- ${target}` : ""}`;
}

export function RevisionPicker({ revisions, value, onChange, exclude = [] }: RevisionPickerProps) {
  const options = revisions.filter((r) => !exclude.includes(r.revision));

  return (
    <select
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
    >
      {options.map((r) => (
        <option key={r.revision} value={r.revision}>
          {labelFor(r)} {r.has_verification_report ? "✓ verified" : ""}
        </option>
      ))}
    </select>
  );
}
