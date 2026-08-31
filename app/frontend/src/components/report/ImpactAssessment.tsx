import type { ImpactAssessment as ImpactAssessmentType } from "../../api/types";

function pct(x: number | null): string {
  return x == null ? "n/a" : `${(x * 100).toFixed(1)}%`;
}

function Badge({ pass, passLabel, failLabel }: { pass: boolean; passLabel: string; failLabel: string }) {
  return <span className={pass ? "badge-pass" : "badge-fail"}>{pass ? passLabel : failLabel}</span>;
}

/**
 * Neighbor/general drift and forget-collapse as colored pass/fail badges (green
 * when *_within_tolerance / forget_collapsed is true, red otherwise) PLUS the
 * underlying before/after numbers -- a judge should see the badge and the number
 * it came from, never just a color.
 */
export function ImpactAssessment({ impact }: { impact: ImpactAssessmentType }) {
  return (
    <section className="card space-y-3">
      <h3 className="font-semibold text-slate-900 dark:text-slate-100">Impact Assessment</h3>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-slate-200 p-3 dark:border-slate-700">
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Forget-set (after)</p>
          <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100">{pct(impact.forget_accuracy_after)}</p>
          <Badge pass={impact.forget_collapsed} passLabel="collapsed (genuinely forgotten)" failLabel="did NOT collapse -- investigate" />
        </div>

        <div className="rounded-md border border-slate-200 p-3 dark:border-slate-700">
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Neighbor accuracy</p>
          <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100">
            {pct(impact.neighbor_accuracy_before)} &rarr; {pct(impact.neighbor_accuracy_after)}
          </p>
          <Badge pass={impact.neighbor_within_tolerance} passLabel="within tolerance" failLabel="drifted beyond tolerance" />
        </div>

        <div className="rounded-md border border-slate-200 p-3 dark:border-slate-700">
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">General-retain accuracy</p>
          <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100">
            {pct(impact.general_accuracy_before)} &rarr; {pct(impact.general_accuracy_after)}
          </p>
          <Badge pass={impact.general_within_tolerance} passLabel="within tolerance" failLabel="drifted beyond tolerance" />
        </div>
      </div>
    </section>
  );
}
