/**
 * TypeScript mirror of Module 5's HTTP-facing shapes. Hand-kept in sync with the
 * Python source rather than generated -- see plan.md's Module 6 "Decisions worth
 * stating outright" for why (a five-endpoint hackathon-timeline API doesn't earn a
 * codegen step yet). Every type below traces back to one of:
 *   - app/backend/schemas.py           (ChatMessage/ChatRequest/ChatResponse,
 *                                        ErasureRequestBody, JobStatus)
 *   - app/backend/manifest_view.py      (RevisionSummary + its "kind"-discriminated
 *                                        accuracy union -- revision-0's eval_summary
 *                                        vs. revision-N's accuracy_before/after are
 *                                        genuinely different measurements, never
 *                                        forced into one shape)
 *   - app/backend/routes/meta.py         (EntityListItem, AttributesResponse,
 *                                        ExampleRequest)
 *   - verification/report.py's build_report() (ErasureReport and everything under
 *     verification_results -- direct_qa.py/relational_probe.py/mia.py/
 *     reference_comparison.py/general_capability.py's actual return shapes)
 * If any of those Python files change shape, this file needs a matching edit --
 * there is no build step that would catch drift automatically yet.
 */

// --- app/backend/schemas.py --------------------------------------------------

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  revision: number;
  messages: ChatMessage[];
  max_new_tokens?: number | null;
}

export interface ChatResponse {
  revision: number;
  adapter_label: string;
  reply: string;
}

export type UnlearningMethod = "npo" | "ga";

export interface ErasureRequestBody {
  entity?: string | null;
  attribute?: string | null;
  method: UnlearningMethod;
  parent_revision?: number | null;
  max_steps?: number | null;
  auto_verify: boolean;
}

export type JobType = "train_and_verify" | "train_only" | "verify_only";
export type JobStatusValue = "queued" | "running" | "verifying" | "done" | "failed";

export interface JobStatus {
  job_id: string;
  job_type: JobType;
  status: JobStatusValue;
  erasure_request?: Record<string, unknown> | null;
  method?: string | null;
  parent_revision?: number | null;
  max_steps?: number | null;
  auto_verify: boolean;
  revision?: number | null;
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  log_tail: string[];
}

export const ACTIVE_JOB_STATUSES: readonly JobStatusValue[] = ["queued", "running", "verifying"];

// --- app/backend/manifest_view.py --------------------------------------------

/** revision-0's shape: a heldout sanity-check accuracy, unrelated to any erasure
 * request -- NOT a forget-accuracy number. */
export interface BaselineEvalSummaryAccuracy {
  kind: "baseline_eval_summary";
  eval_summary: Record<string, unknown>;
  headline_accuracy: number | null;
}

/** revision-N's shape: forget/neighbor/general/forget_probe accuracy for THAT
 * specific erasure request, before and after unlearning. */
export interface UnlearningBeforeAfterAccuracy {
  kind: "unlearning_accuracy_before_after";
  accuracy_before: ThreeWayAccuracy | null;
  accuracy_after: ThreeWayAccuracy | null;
  early_stop_step: number | null;
  headline_accuracy: number | null;
}

export type RevisionAccuracy = BaselineEvalSummaryAccuracy | UnlearningBeforeAfterAccuracy;

export interface RevisionSummary {
  revision: number;
  label: string | null;
  parent_revision: number | null;
  method: string | null;
  erasure_request: Record<string, unknown> | null;
  base_model: string | null;
  adapter_path: string | null;
  lora_config: Record<string, unknown> | null;
  training_args: Record<string, unknown> | null;
  created_at: string | null;
  accuracy: RevisionAccuracy;
  has_verification_report: boolean;
}

// --- app/backend/routes/meta.py ----------------------------------------------

export interface EntityListItem {
  entity: string;
  entity_type: string;
  fact_group_id: string;
}

export interface AttributesResponse {
  company: string[];
  person: string[];
}

export interface ExampleRequest {
  name: string;
  entity: string | null;
  attribute: string | null;
  comment: string | null;
}

// --- verification/*.py's signal shapes (consumed via verification/report.py) --

/** verification/direct_qa.py's accuracy_on() summary -- overall_accuracy is null
 * (not 0.0) when a pool has zero scoreable records, per that function's own
 * "no signal vs. measured zero" distinction. */
export interface AccuracySummary {
  n: number;
  overall_accuracy: number | null;
  accuracy_by_attribute: Record<string, number>;
}

/** verification/direct_qa.py's three_way_accuracy() -- the shape of both
 * direct_qa_before and direct_qa_after in a report. */
export interface ThreeWayAccuracy {
  forget: AccuracySummary;
  retain_neighbor: AccuracySummary;
  retain_general_unrelated: AccuracySummary;
  forget_probe: AccuracySummary;
  forget_details: Record<string, unknown>[];
  forget_probe_details: Record<string, unknown>[];
}

/** verification/relational_probe.py's probe_relational(). */
export interface RelationalProbeResult {
  summary: {
    n_relational_examples_probed: number;
    forgotten_entity_dropped_out_rate: number | null;
    retained_sibling_preserved_rate: number | null;
  };
  details: Record<string, unknown>[];
}

/** verification/relational_probe.py's check_decoys() -- one entry per matched
 * decoy fact (or a "no_matching_fact_found" / "no_forward_qa_record_in_train_jsonl"
 * status entry when the declared check couldn't even be probed). */
export interface DecoyCheckResult {
  status?: "no_matching_fact_found" | "no_forward_qa_record_in_train_jsonl";
  probed_entity?: string;
  expected_value?: string;
  question?: string;
  generated_answer?: string;
  correct?: boolean;
  [key: string]: unknown; // spreads verification/config.py's DECOY_CHECKS entry fields too
}

/** verification/mia.py's run_mia(). */
export interface MiaResult {
  summary: {
    mean_percentile_rank: number | null;
    small_forget_set_caveat: boolean;
  };
  per_example_ranks: number[];
}

/** verification/reference_comparison.py's compare_against_reference() -- a
 * discriminated union on `available`; NEVER a silently omitted section, per that
 * module's own docstring, even when no reference model exists for this request. */
export type ReferenceComparisonResult =
  | { available: false; reason: string }
  | {
      available: true;
      reference_entity: string;
      reference_fact_group_id: string;
      reference_adapter_path: string;
      scoped_to_n_forget_records: number;
      reference_forget_set_accuracy: AccuracySummary;
      reference_forget_set_details: Record<string, unknown>[];
      note: string;
    };

/** verification/general_capability.py's run_general_capability(). */
export interface GeneralCapabilityResult {
  generic_prompts: {
    summary: { n: number; n_correct: number; accuracy: number | null };
    details: Record<string, unknown>[];
  };
  previous_company_control_group: {
    summary: AccuracySummary;
    details: Record<string, unknown>[];
  };
}

/** verification/report.py's build_impact_assessment(). */
export interface ImpactAssessment {
  neighbor_accuracy_before: number | null;
  neighbor_accuracy_after: number | null;
  neighbor_drift: number | null;
  neighbor_within_tolerance: boolean;
  general_accuracy_before: number | null;
  general_accuracy_after: number | null;
  general_drift: number | null;
  general_within_tolerance: boolean;
  forget_accuracy_after: number | null;
  forget_collapsed: boolean;
}

export interface MethodComparison {
  this_method: string;
  sibling_method: string;
  sibling_revision: number;
  this_forget_accuracy: number | null;
  sibling_forget_accuracy: number | null;
  this_neighbor_accuracy: number | null;
  sibling_neighbor_accuracy: number | null;
  [key: string]: unknown;
}

export interface VerificationResultsSection {
  direct_qa_before: ThreeWayAccuracy;
  direct_qa_after: ThreeWayAccuracy;
  relational_probing: RelationalProbeResult;
  decoy_checks: DecoyCheckResult[];
  membership_inference: MiaResult;
  reference_model_comparison: ReferenceComparisonResult;
  general_capability: GeneralCapabilityResult;
  npo_vs_ga_comparison: MethodComparison | null;
}

/** verification/report.py's build_report() -- the full Erasure Report, exactly as
 * written to verification/reports/revision-<N>_verification_report.json and served
 * unmodified by GET /reports/{revision}. The brief's own five sections: What Was
 * Targeted, What Was Done, Verification Results, Impact Assessment, Key Takeaways. */
/** verification/run_verification.py's resolved_summary, spread into
 * what_was_targeted alongside erasure_request -- exactly what got resolved from
 * the request (unlearning/selectors.py's ResolvedRequest), not a re-derivation. */
export interface ResolvedRequestSummary {
  request_type: "entity" | "attribute_cell" | "attribute_type";
  entity_type: string | null;
  forget_fact_ids: string[];
  forget_fact_group_ids: string[];
  retain_neighbor_entities: string[];
}

export interface ErasureReport {
  revision: number;
  method: string;
  created_at: string;
  what_was_targeted: ResolvedRequestSummary & {
    /** unlearning/request.py's ErasureRequest.to_dict(): {entity, attribute, request_type}. */
    erasure_request: { entity: string | null; attribute: string | null; request_type: string };
  };
  what_was_done: {
    method: string;
    parent_revision: number | null;
    training_args: Record<string, unknown>;
    early_stop_step: number | null;
  };
  verification_results: VerificationResultsSection;
  impact_assessment: ImpactAssessment;
  key_takeaways: string[];
}

// --- Shared HTTP error shape ---------------------------------------------------

/** Every app/backend HTTPException body is {"detail": "..."} -- FastAPI's default
 * error envelope, never a raw traceback (see main.py's startup checks / adapters.py's
 * HeavyDepsMissing). */
export interface ApiErrorBody {
  detail: string;
}
