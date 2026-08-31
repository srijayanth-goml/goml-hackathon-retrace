import type {
  AccuracySummary,
  DecoyCheckResult,
  ThreeWayAccuracy,
  VerificationResultsSection,
} from "../../api/types";

function pct(x: number | null): string {
  return x == null ? "n/a" : `${(x * 100).toFixed(1)}%`;
}

function ThreeWayTable({ before, after }: { before: ThreeWayAccuracy; after: ThreeWayAccuracy }) {
  const rows: { label: string; key: keyof ThreeWayAccuracy }[] = [
    { label: "Forget set", key: "forget" },
    { label: "Retain (neighbor)", key: "retain_neighbor" },
    { label: "Retain (general/unrelated)", key: "retain_general_unrelated" },
    { label: "Forget probe (never trained against)", key: "forget_probe" },
  ];

  return (
    <table className="min-w-full text-sm">
      <thead>
        <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:text-slate-400">
          <th className="py-1 pr-4">Pool</th>
          <th className="py-1 pr-4">Before</th>
          <th className="py-1 pr-4">After</th>
          <th className="py-1 pr-4">n (after)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const b = before[row.key] as AccuracySummary;
          const a = after[row.key] as AccuracySummary;
          return (
            <tr key={row.key} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
              <td className="py-1 pr-4 text-slate-700 dark:text-slate-200">{row.label}</td>
              <td className="py-1 pr-4 text-slate-600 dark:text-slate-300">{pct(b.overall_accuracy)}</td>
              <td className="py-1 pr-4 font-medium text-slate-900 dark:text-slate-100">{pct(a.overall_accuracy)}</td>
              <td className="py-1 pr-4 text-slate-500 dark:text-slate-400">{a.n}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function DecoyChecks({ decoys }: { decoys: DecoyCheckResult[] }) {
  if (decoys.length === 0) return <p className="text-sm text-slate-500 dark:text-slate-400">No decoy checks configured.</p>;
  return (
    <ul className="space-y-1 text-sm">
      {decoys.map((d, i) => {
        const notProbed = d.status === "no_matching_fact_found" || d.status === "no_forward_qa_record_in_train_jsonl";
        return (
          <li key={i} className="flex items-center gap-2">
            {notProbed ? (
              <span className="badge-neutral">{d.status}</span>
            ) : (
              <span className={d.correct ? "badge-pass" : "badge-fail"}>{d.correct ? "correctly retained" : "over-forgotten"}</span>
            )}
            <span className="text-slate-700 dark:text-slate-200">
              {d.probed_entity ? `${d.probed_entity}` : "(entity not resolved)"}
              {typeof d.check_attribute === "string" ? ` -- ${d.check_attribute}` : ""}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Direct QA before/after, relational probing, decoy checks, MIA, reference-model
 * comparison, general capability, and NPO-vs-GA comparison -- every sub-signal
 * verification/report.py's build_report() writes under verification_results.
 * Nothing here is summarized away: reference_model_comparison's `available: false`
 * gets an explicit "not available for this request" render, never a blank section,
 * and MIA's small_forget_set_caveat is shown, not hidden.
 */
export function VerificationResults({ vr }: { vr: VerificationResultsSection }) {
  return (
    <section className="card space-y-5">
      <h3 className="font-semibold text-slate-900 dark:text-slate-100">Verification Results</h3>

      <div>
        <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">Direct QA (forward + paraphrase + reverse)</h4>
        <ThreeWayTable before={vr.direct_qa_before} after={vr.direct_qa_after} />
      </div>

      <div>
        <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">Multi-hop relational probing</h4>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          {vr.relational_probing.summary.n_relational_examples_probed} example(s) probed -- forgotten entity dropped
          out of {pct(vr.relational_probing.summary.forgotten_entity_dropped_out_rate)} of relational answers;
          retained siblings preserved in {pct(vr.relational_probing.summary.retained_sibling_preserved_rate)}.
        </p>
      </div>

      <div>
        <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">Decoy / over-forgetting checks</h4>
        <DecoyChecks decoys={vr.decoy_checks} />
      </div>

      <div>
        <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">Membership inference (loss-based)</h4>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Mean percentile rank vs. never-seen text:{" "}
          {vr.membership_inference.summary.mean_percentile_rank != null
            ? vr.membership_inference.summary.mean_percentile_rank.toFixed(2)
            : "n/a"}
          {vr.membership_inference.summary.small_forget_set_caveat && (
            <span className="badge-neutral ml-2">small forget-set caveat applies</span>
          )}
        </p>
      </div>

      <div>
        <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">Reference ("never learned it") model comparison</h4>
        {vr.reference_model_comparison.available ? (
          <p className="text-sm text-slate-600 dark:text-slate-300">
            {vr.reference_model_comparison.reference_entity} scores{" "}
            {pct(vr.reference_model_comparison.reference_forget_set_accuracy.overall_accuracy)} on the same
            forgotten facts ({vr.reference_model_comparison.scoped_to_n_forget_records} record(s) scoped).
          </p>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Not available for this request: {vr.reference_model_comparison.reason}
          </p>
        )}
      </div>

      <div>
        <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">General-capability spot-check</h4>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Generic prompts: {pct(vr.general_capability.generic_prompts.summary.accuracy)} on{" "}
          {vr.general_capability.generic_prompts.summary.n} prompt(s). Previous-company control group:{" "}
          {pct(vr.general_capability.previous_company_control_group.summary.overall_accuracy)}.
        </p>
      </div>

      {vr.npo_vs_ga_comparison && (
        <div>
          <h4 className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-200">
            {vr.npo_vs_ga_comparison.this_method.toUpperCase()} vs. {vr.npo_vs_ga_comparison.sibling_method.toUpperCase()}
          </h4>
          <p className="text-sm text-slate-600 dark:text-slate-300">
            Forget accuracy {pct(vr.npo_vs_ga_comparison.this_forget_accuracy)} vs.{" "}
            {pct(vr.npo_vs_ga_comparison.sibling_forget_accuracy)}; neighbor accuracy{" "}
            {pct(vr.npo_vs_ga_comparison.this_neighbor_accuracy)} vs.{" "}
            {pct(vr.npo_vs_ga_comparison.sibling_neighbor_accuracy)} (revision-{vr.npo_vs_ga_comparison.sibling_revision}).
          </p>
        </div>
      )}
    </section>
  );
}
