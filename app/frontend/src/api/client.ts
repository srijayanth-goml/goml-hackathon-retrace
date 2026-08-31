/**
 * Thin fetch wrapper over Module 5's API. One typed function per endpoint (per
 * plan.md's Module 6 Step 2) so a route's request/response shape is checked at
 * compile time, and a non-2xx response is always a typed ApiError -- FastAPI's
 * HTTPException body is always {"detail": "..."} (schemas.py has no other error
 * envelope), so parsing it is a one-shape operation, not a guess.
 */
import type {
  AttributesResponse,
  ChatRequest,
  ChatResponse,
  EntityListItem,
  ErasureReport,
  ErasureRequestBody,
  ExampleRequest,
  JobStatus,
  RevisionSummary,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

/** Thrown for every non-2xx response. `status` lets callers branch on 503
 * (HeavyDepsMissing -- torch/transformers/peft not installed on the backend
 * machine) vs. 400/404 (a normal, expected rejection) without string-matching. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

/** True for a 503 specifically -- app/backend/adapters.py's HeavyDepsMissing,
 * raised by any route that touches the model when torch/transformers/peft aren't
 * installed. Kept as a named check rather than `err.status === 503` scattered
 * across components, so the "what does 503 mean here" reasoning lives in one place. */
export function isHeavyDepsMissing(err: unknown): err is ApiError {
  return err instanceof ApiError && err.status === 503;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    // fetch itself throwing (network down, backend not running, CORS rejected
    // pre-flight) -- not an HTTP status at all, so it doesn't fit ApiError's
    // status-code contract. Callers see this as a plain Error with a message a
    // judge-facing banner can still display.
    throw new Error(
      `Could not reach the ReTrace backend at ${API_BASE}. Is \`uvicorn app.backend.main:app\` running?`,
    );
  }

  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // non-JSON error body -- keep the statusText fallback above
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 202 || res.status === 204) {
    // POST /erasure-requests and POST /reports/{revision}/generate both return
    // 202 with a JobStatus body (jobs.submit_*_job's return value) -- still JSON,
    // handled the same as 200 below. 204 has no body; T should be `void` there.
    const text = await res.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  return (await res.json()) as T;
}

// --- routes/revisions.py ------------------------------------------------------

export function getRevisions(): Promise<RevisionSummary[]> {
  return request<RevisionSummary[]>("/revisions");
}

export function getRevision(revision: number): Promise<RevisionSummary> {
  return request<RevisionSummary>(`/revisions/${revision}`);
}

// --- routes/chat.py ------------------------------------------------------------

export function postChat(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) });
}

// --- routes/erasure_requests.py -------------------------------------------------

export function postErasureRequest(body: ErasureRequestBody): Promise<JobStatus> {
  return request<JobStatus>("/erasure-requests", { method: "POST", body: JSON.stringify(body) });
}

export function listJobs(): Promise<JobStatus[]> {
  return request<JobStatus[]>("/jobs");
}

export function getJob(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/jobs/${jobId}`);
}

// --- routes/reports.py -----------------------------------------------------------

export function getReport(revision: number): Promise<ErasureReport> {
  return request<ErasureReport>(`/reports/${revision}`);
}

export function getReportMarkdown(revision: number): Promise<string> {
  // PlainTextResponse on the backend, not JSON -- bypass request()'s res.json().
  return fetch(`${API_BASE}/reports/${revision}/markdown`).then(async (res) => {
    if (!res.ok) {
      const detail = await res.text();
      throw new ApiError(res.status, detail || res.statusText);
    }
    return res.text();
  });
}

export function postGenerateReport(revision: number): Promise<JobStatus> {
  return request<JobStatus>(`/reports/${revision}/generate`, { method: "POST" });
}

// --- routes/meta.py ---------------------------------------------------------------

export function getEntities(): Promise<EntityListItem[]> {
  return request<EntityListItem[]>("/entities");
}

export function getAttributes(): Promise<AttributesResponse> {
  return request<AttributesResponse>("/attributes");
}

export function getExampleRequests(): Promise<ExampleRequest[]> {
  return request<ExampleRequest[]>("/requests/examples");
}
