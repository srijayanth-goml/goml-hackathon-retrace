# Module 5 -- App Backend

Serves the live model behind an API (Design Doc Section 8): query the current
(post-erasure) model, submit a new erasure request and watch it run through Module
3 then Module 4, list/compare manifest revisions, fetch a generated Erasure Report.
This module is thin by design -- it orchestrates Modules 2-4's own code and reports,
it never reimplements targeting, unlearning, or verification. See `../../plan.md`'s
"Module 5 -- App Backend -- detailed plan" for the full design and the reasoning
behind each decision.

## Layout

- `config.py` -- host/port, generation defaults (greedy decode, a server-side
  `max_new_tokens` cap), job-queue paths.
- `schemas.py` -- pydantic request/response models for the HTTP layer.
- `manifest_view.py` -- normalizes `finetuning/checkpoints/manifest.json`'s
  revision-0 shape (`eval_summary`) and revision-N shape (`accuracy_before`/
  `accuracy_after`/`early_stop_step`) into one shared shape, rather than asking the
  frontend to special-case revision 0.
- `adapters.py` -- `AdapterCache`: loads the base model ONCE, attaches every
  revision's adapter as a NAMED peft adapter on it, switches with `set_adapter()`
  before generating. Avoids reloading the full base model per revision switch, which
  matters on the laptop this runs on (see "A hardware fact" in the plan). Exposes
  `MODEL_LOCK`, shared with `jobs.py` so a chat request and a training/verification
  job never touch the model at the same moment.
- `inference.py` -- multi-turn `apply_chat_template` generation for the chat UI,
  kept separate from `finetuning/eval_quick.py`'s single-turn, exact-match-graded
  `generate_answer`.
- `jobs.py` -- a single background worker thread + `jobs.json` history. Runs
  `unlearning.train.run(...)` then, if `auto_verify`, `verification.run_verification.
  run(revision)`, both under `MODEL_LOCK`.
- `routes/` -- `revisions.py`, `chat.py`, `erasure_requests.py`, `reports.py`,
  `meta.py`.
- `main.py` -- the FastAPI app: `uvicorn app.backend.main:app --reload`.
- `jobs/jobs.json` -- persisted job history (gitignored).
- `tests/` -- see below.

## Run it

Needs `fastapi`/`uvicorn`/`pydantic`/`httpx` (added to the repo-ROOT
`requirements.txt`, not a separate `app/backend/requirements.txt` -- this module
runs locally like `data_pipeline`/`unlearning`/`verification` do, so it shares their
file rather than starting a fourth one -- locked decision, see `../../plan.md`'s
Module 5 Open Decisions) plus `torch`/`transformers`/`peft` for the model-dependent
routes (already in the same file for Module 3):

```bash
pip install -r requirements.txt
uvicorn app.backend.main:app --reload
```

The server boots and every route that doesn't touch the model works even before
`torch`/`transformers`/`peft` are installed -- `POST /chat` and a real erasure-request
job return a clean `503` / a `"failed"` job status with a clear `error` message
instead, never a raw traceback (`app/backend/adapters.py`'s `HeavyDepsMissing`).

Key endpoints:

- `GET /revisions`, `GET /revisions/{n}` -- the normalized manifest.
- `POST /chat` `{revision, messages, max_new_tokens?}` -- live chat against any
  revision. The same request shape against `revision=0` and a later revision is the
  concrete proof of Design Doc Section 8's "compare live, not a recorded demo."
- `POST /erasure-requests` `{entity?, attribute?, method, parent_revision?,
  max_steps?, auto_verify}` -- validated synchronously (torch-free) before
  enqueueing; returns a `job_id`.
- `GET /jobs`, `GET /jobs/{job_id}` -- job status/history.
- `GET /reports/{revision}`, `GET /reports/{revision}/markdown`,
  `POST /reports/{revision}/generate` -- the Erasure Report, as Module 4 wrote it.
- `GET /entities`, `GET /attributes`, `GET /requests/examples` -- backs a picker UI
  instead of a freeform entity-name text box.

## Run the tests

```bash
pytest app/backend/tests
```

Every route not requiring an actually-loaded model is tested for real here
(manifest normalization against the real file, request validation against the real
neighbor lookup -- including the real heldout-entity regression case, `Silvergate
Aerospace` -- job-store transitions with a fake job function, route smoke tests).
`test_request_validation.py`'s last test submits a genuinely valid request through
the real queue/worker thread with no mocking; what happens next depends on whether
`torch`/`transformers`/`peft` are installed in the environment running the tests --
either a real reply, or a job that fails cleanly with a `"torch"/"transformers"/
"peft"` message, never a hang or a crash.

## Locked decisions (see `../../plan.md`'s Module 5 section for the reasoning)

- `fastapi`/`uvicorn`/`pydantic`/`httpx` live in the repo-root `requirements.txt`,
  continuing the pattern `data_pipeline`/`unlearning`/`verification` already
  established for modules that run on the same local machine.
- One base model, N named peft adapters (`model.load_adapter(..., adapter_name=...)`
  + `model.set_adapter(...)`), not a reload per revision switch.
- One global lock (`adapters.MODEL_LOCK`) serializes every chat generation call and
  every training/verification job -- nothing runs at the same time as anything else,
  given the actual hardware this runs on.
- `unlearning/config.py`'s `DEFAULT_PARENT_REVISION = 0` stays the default; the API
  accepts an optional `parent_revision` override rather than making sequential
  composition the accidental default.
- Chat is stateless: the caller sends the full `messages` array every time, no
  server-side session store.
- Job history is a flat `jobs.json`, not a database.

## Open items still to confirm

- Whether `peft`'s named multi-adapter switching behaves cleanly against this exact
  checkpoint shape once actually exercised with a real revision-1 -- the fallback
  (a small reload-per-switch cache) is a contained change to `adapters.py` alone if
  not.
- Whether the single-global-lock design leaves the chat experience feeling too slow
  during a live training/verification run for a judge -- acceptable trade for
  correctness on this hardware, revisit only if a real run shows otherwise.
