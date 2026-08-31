import type { ErasureReport } from "../../api/types";

export function WhatWasTargeted({ report }: { report: ErasureReport }) {
  const t = report.what_was_targeted;
  return (
    <section className="card space-y-2">
      <h3 className="font-semibold text-slate-900 dark:text-slate-100">What Was Targeted</h3>
      <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-sm sm:grid-cols-2">
        <dt className="text-slate-500 dark:text-slate-400">Entity</dt>
        <dd className="text-slate-800 dark:text-slate-100">{t.erasure_request.entity ?? "(none -- attribute-type request)"}</dd>
        <dt className="text-slate-500 dark:text-slate-400">Attribute</dt>
        <dd className="text-slate-800 dark:text-slate-100">{t.erasure_request.attribute ?? "(none -- whole-entity request)"}</dd>
        <dt className="text-slate-500 dark:text-slate-400">Request type</dt>
        <dd className="text-slate-800 dark:text-slate-100">{t.request_type}</dd>
        <dt className="text-slate-500 dark:text-slate-400">Entity type</dt>
        <dd className="text-slate-800 dark:text-slate-100">{t.entity_type ?? "n/a"}</dd>
        <dt className="text-slate-500 dark:text-slate-400">Forgotten facts</dt>
        <dd className="text-slate-800 dark:text-slate-100">{t.forget_fact_ids.length} fact(s) across {t.forget_fact_group_ids.length} entity group(s)</dd>
        <dt className="text-slate-500 dark:text-slate-400">Confusable neighbors sampled into retain</dt>
        <dd className="text-slate-800 dark:text-slate-100">
          {t.retain_neighbor_entities.length > 0 ? t.retain_neighbor_entities.join(", ") : "none"}
        </dd>
      </dl>
    </section>
  );
}
