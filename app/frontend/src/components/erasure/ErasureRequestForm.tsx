import { useMemo, useState } from "react";
import { postErasureRequest } from "../../api/client";
import { useActiveJob } from "../../hooks/useActiveJob";
import { useMeta } from "../../hooks/useMeta";
import { useHeavyDeps } from "../layout/HeavyDepsContext";
import { ExampleRequestPicker } from "./ExampleRequestPicker";
import { JobStatusPanel } from "./JobStatusPanel";
import type { ExampleRequest, JobStatus, UnlearningMethod } from "../../api/types";

const NONE = "";

interface ErasureRequestFormProps {
  onJobDone?: (job: JobStatus) => void;
}

/**
 * Entity dropdown + attribute dropdown (unlearning/request.py's ErasureRequest
 * needs at least one of the two -- entity-only, attribute-only, or both, the three
 * request types Design Doc Section 3 defines), method radio, auto_verify checkbox,
 * example-request quick-fill, and the live job status once submitted. Submission
 * itself is disabled while ANY job is active (useActiveJob) -- app/backend runs one
 * job at a time on one lock, so a second submission would just queue silently
 * behind the first with no way for this screen to show its status separately.
 */
export function ErasureRequestForm({ onJobDone }: ErasureRequestFormProps) {
  const { entities, attributes, examples, loading, error: metaError } = useMeta();
  const { activeJob } = useActiveJob();
  const { reportIfHeavyDepsMissing } = useHeavyDeps();

  const [entity, setEntity] = useState(NONE);
  const [attribute, setAttribute] = useState(NONE);
  const [method, setMethod] = useState<UnlearningMethod>("npo");
  const [autoVerify, setAutoVerify] = useState(true);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);

  const selectedEntityType = useMemo(
    () => entities.find((e) => e.entity === entity)?.entity_type ?? null,
    [entities, entity],
  );

  const attributeOptions = useMemo(() => {
    if (selectedEntityType === "company") return attributes.company;
    if (selectedEntityType === "person") return attributes.person;
    // No entity chosen (or an attribute-type request in progress): offer the union
    // -- an attribute-type request isn't scoped to one entity type.
    return [...new Set([...attributes.company, ...attributes.person])].sort();
  }, [attributes, selectedEntityType]);

  const canSubmit = (entity !== NONE || attribute !== NONE) && !submitting && !activeJob;

  const applyExample = (ex: ExampleRequest) => {
    setEntity(ex.entity ?? NONE);
    setAttribute(ex.attribute ?? NONE);
  };

  const submit = () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setSubmitError(null);
    postErasureRequest({
      entity: entity || null,
      attribute: attribute || null,
      method,
      auto_verify: autoVerify,
    })
      .then((job) => setJobId(job.job_id))
      .catch((err: unknown) => {
        reportIfHeavyDepsMissing(err);
        setSubmitError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setSubmitting(false));
  };

  return (
    <div className="space-y-4">
      <div className="card space-y-3">
        <h3 className="font-semibold text-slate-900 dark:text-slate-100">Example requests</h3>
        {loading && <p className="text-sm text-slate-400">Loading entities/attributes...</p>}
        {metaError && <p className="text-sm text-rose-600 dark:text-rose-400">{metaError}</p>}
        <ExampleRequestPicker examples={examples} onPick={applyExample} disabled={!!activeJob} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="card space-y-3"
      >
        <h3 className="font-semibold text-slate-900 dark:text-slate-100">Submit an erasure request</h3>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-sm text-slate-600 dark:text-slate-300">
            Entity
            <select
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
              disabled={!!activeJob}
              className="rounded-md border border-slate-300 px-2 py-1 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            >
              <option value={NONE}>-- none (attribute-type request) --</option>
              {entities.map((e) => (
                <option key={e.entity} value={e.entity}>
                  {e.entity} ({e.entity_type})
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm text-slate-600 dark:text-slate-300">
            Attribute
            <select
              value={attribute}
              onChange={(e) => setAttribute(e.target.value)}
              disabled={!!activeJob}
              className="rounded-md border border-slate-300 px-2 py-1 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            >
              <option value={NONE}>-- none (whole-entity request) --</option>
              {attributeOptions.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <fieldset className="flex items-center gap-3 text-sm text-slate-600 dark:text-slate-300">
            <legend className="sr-only">Method</legend>
            <label className="flex items-center gap-1">
              <input
                type="radio"
                name="method"
                value="npo"
                checked={method === "npo"}
                onChange={() => setMethod("npo")}
                disabled={!!activeJob}
              />
              NPO (primary)
            </label>
            <label className="flex items-center gap-1">
              <input
                type="radio"
                name="method"
                value="ga"
                checked={method === "ga"}
                onChange={() => setMethod("ga")}
                disabled={!!activeJob}
              />
              Gradient Ascent (comparison baseline)
            </label>
          </fieldset>

          <label className="flex items-center gap-1 text-sm text-slate-600 dark:text-slate-300">
            <input
              type="checkbox"
              checked={autoVerify}
              onChange={(e) => setAutoVerify(e.target.checked)}
              disabled={!!activeJob}
            />
            Auto-verify after training
          </label>
        </div>

        {activeJob && (
          <p className="text-xs text-sky-700 dark:text-sky-400">
            A job is already running -- submission is disabled until it finishes (app/backend runs one job at a
            time on one model lock).
          </p>
        )}
        {submitError && <p className="text-sm text-rose-600 dark:text-rose-400">{submitError}</p>}

        <button type="submit" disabled={!canSubmit} className="btn">
          {submitting ? "Submitting..." : "Submit erasure request"}
        </button>
      </form>

      {jobId && <JobStatusPanel jobId={jobId} onSettled={onJobDone} />}
    </div>
  );
}
