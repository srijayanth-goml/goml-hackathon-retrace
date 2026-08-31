# Module 6 -- App UI

The judge-facing frontend (Design Doc Section 8 / the brief's "UI / Demo Frontend
Stack of your choice", "keep the model live and queryable at judging time"): live
chat against any manifest revision with revision-0 pinned side by side for a real
before/after, a form to submit or pick an erasure request and watch it run, a
revision list, and a renderer for Module 4's five-section Erasure Report. Talks only
to Module 5's API (`app/backend`) -- see `../../plan.md`'s "Module 6 -- App UI --
detailed plan" for the full design and the reasoning behind each decision.

## Stack

React + TypeScript + Vite + Tailwind CSS. No router (four tabs in one `App.tsx`, no
deep-linking need for a local judging demo), no state-management or data-fetching
library (a handful of custom hooks over `fetch` cover this app's whole data surface),
polling (not a websocket) for job status, since Module 5 only exposes
`GET /jobs/{job_id}`. See the plan's "Framework and stack" section for why -- short
version: `app/backend/config.py`'s CORS allowlist already had Vite's default
`:5173` port whitelisted before this module existed.

## Layout

- `src/api/` -- `types.ts` (hand-kept TS mirror of `app/backend/schemas.py`,
  `manifest_view.py`, and `verification/report.py`'s `build_report()` shape) and
  `client.ts` (a typed fetch wrapper, one function per endpoint, a typed `ApiError`
  on any non-2xx response).
- `src/hooks/` -- `useRevisions` (GET /revisions + a `refresh()` called after any
  job completes), `useMeta` (GET /entities, /attributes, /requests/examples),
  `useJobPolling` (polls a job until it's done/failed), `useActiveJob` (is ANY job
  currently running -- backs the single-worker-lock banner and disables a second
  submission).
- `src/components/layout/` -- `TabNav`, `HeavyDepsBanner` + `HeavyDepsContext` (an
  app-wide 503/HeavyDepsMissing banner), `ActiveJobBanner`.
- `src/components/chat/` -- `ChatPane` (one revision's conversation), `CompareChat`
  (two panes side by side, left pinned to revision-0, a "send to both" input),
  `RevisionPicker`.
- `src/components/erasure/` -- `ErasureRequestForm`, `ExampleRequestPicker` (the
  canned demo requests from `unlearning/requests/`), `JobStatusPanel`.
- `src/components/revisions/` -- `RevisionList`.
- `src/components/report/` -- `ReportView` plus one component per Erasure Report
  section (`WhatWasTargeted`, `WhatWasDone`, `VerificationResults`,
  `ImpactAssessment`, `KeyTakeaways`) and `RawReportJson` -- a collapsed raw-JSON
  fallback that stays present regardless of how complete the structured rendering
  is, so nothing Module 4 computes is ever actually hidden from a judge.

## Run it

```bash
npm install
cp .env.example .env.local   # only needed if your backend isn't on 127.0.0.1:8000
npm run dev
```

Needs Module 5's API running first:

```bash
# from the repo root, in a separate terminal
pip install -r requirements.txt
uvicorn app.backend.main:app --reload
```

The dev server serves on `http://localhost:5173` (Vite's default -- already in
`app/backend/config.py`'s `CORS_ALLOW_ORIGINS`, confirmed live: a CORS preflight
against a running backend returns `access-control-allow-origin: http://localhost:5173`
with no changes needed on either side).

`npm run build` produces `dist/` for a static deploy; `npm run preview` serves that
build locally.

## Run the tests / typecheck

```bash
npm run typecheck   # tsc -b --noEmit
npm run test        # vitest run
```

16 tests pass today with no backend running at all: `api/client.test.ts` (error
handling -- a 400 and a 503 both produce the right typed `ApiError`, a network
failure produces a plain `Error` naming the backend), `useJobPolling.test.ts` (fake
timers -- polls while running/verifying, stops exactly once on done/failed, never
leaks an interval), `ErasureRequestForm.test.tsx` (submit disabled with neither
entity nor attribute chosen, an example-request click pre-fills the form, submit
disabled while a job is active), and `ReportView.test.tsx` (renders every one of the
five report sections against `tests/fixtures/report.revision-1.json`).

That fixture is **hand-built today**, not captured from a real run -- Module 3
hasn't produced a real `revision-1` yet at the time this module was built. It
deliberately covers the two edge cases a naive renderer would silently break on:
`reference_model_comparison.available: false` and
`membership_inference.summary.small_forget_set_caveat: true`. Once Module 3/4
produce a real unlearned, verified revision, copy the real
`verification/reports/revision-1_verification_report.json` over this fixture and
re-run the tests -- same "fixture'd from the real file, not hand-typed" convention
`app/backend/tests/test_manifest_view.py` already established for the manifest.

## Implementation status

Built and passing end to end against a **real** (if untrained) Module 5 instance:
`npm run typecheck`, `npm run test` (16/16), and `npm run build` all pass, and a
live smoke test against a running `uvicorn app.backend.main:app` confirmed --

- `GET /revisions`, `/entities` (100 real entities), `/attributes`, and
  `/requests/examples` (5 real canned requests) all match `src/api/types.ts`
  exactly, field for field -- the types were written from reading
  `app/backend/schemas.py` / `manifest_view.py` / `routes/meta.py` directly, then
  checked against the real running server's actual JSON.
- The CORS preflight for `http://localhost:5173` succeeds against the real backend
  with no configuration changes on either side, confirming the "Vite's default port
  was already whitelisted" assumption the framework decision rested on.
- `POST /chat` against an environment with no `torch`/`transformers`/`peft`
  installed returns the real `503` with `adapters.py`'s actual `HeavyDepsMissing`
  message, which `isHeavyDepsMissing()`/`HeavyDepsBanner` correctly recognize.
- `POST /erasure-requests` against a bad entity returns the real `400` with
  `unlearning/data.py`'s own validation message, unchanged, confirming
  `ErasureRequestForm`'s error path renders the backend's actual detail string.

**Not yet exercised for real:** `torch`/`transformers`/`peft` are not installed in
the environment this module was built in (a deliberate choice -- pulling several GB
of weights into a throwaway smoke-test venv wasn't worth it for validating the
frontend), so `POST /chat`'s success path, a real end-to-end erasure-request job,
and `ReportView` against a real captured report have only been exercised against
mocks/fixtures so far. This is the same posture every other module in this repo was
in before its own first real run -- see `../backend/README.md`'s and
`../../unlearning/README.md`'s own "Implementation status" notes for the pattern.
Re-run the manual smoke test in `../../plan.md`'s Module 6 "Definition of done" once
a real `revision-1` exists.
