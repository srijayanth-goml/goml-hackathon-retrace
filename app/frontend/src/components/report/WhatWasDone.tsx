import type { ErasureReport } from "../../api/types";

export function WhatWasDone({ report }: { report: ErasureReport }) {
  const d = report.what_was_done;
  return (
    <section className="card space-y-2">
      <h3 className="font-semibold text-slate-900 dark:text-slate-100">What Was Done</h3>
      <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-sm sm:grid-cols-2">
        <dt className="text-slate-500 dark:text-slate-400">Method</dt>
        <dd className="text-slate-800 dark:text-slate-100">{d.method === "npo" ? "NPO + neighbor-weighted retain" : d.method.toUpperCase()}</dd>
        <dt className="text-slate-500 dark:text-slate-400">Branched from</dt>
        <dd className="text-slate-800 dark:text-slate-100">{d.parent_revision != null ? `revision-${d.parent_revision}` : "n/a"}</dd>
        <dt className="text-slate-500 dark:text-slate-400">Early-stopped at step</dt>
        <dd className="text-slate-800 dark:text-slate-100">{d.early_stop_step ?? "did not early-stop"}</dd>
      </dl>
      {d.training_args && Object.keys(d.training_args).length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-slate-500 dark:text-slate-400">Training args</summary>
          <pre className="mt-1 overflow-x-auto rounded-md bg-slate-50 p-2 text-xs dark:bg-slate-950/40">
            {JSON.stringify(d.training_args, null, 2)}
          </pre>
        </details>
      )}
    </section>
  );
}
