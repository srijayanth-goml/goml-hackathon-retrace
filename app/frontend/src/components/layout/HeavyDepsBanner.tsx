import { useHeavyDeps } from "./HeavyDepsContext";

/**
 * Shown app-wide the moment any call surfaces a 503 -- explains that
 * torch/transformers/peft aren't installed on the machine running app/backend,
 * rather than leaving a judge staring at a chat box or a stuck job that just never
 * replies. Module 5's own design treats this as an EXPECTED state (every route that
 * touches the model raises it cleanly, per adapters.py's HeavyDepsMissing), so the
 * UI should say so plainly too, not hide it behind a generic error toast.
 */
export function HeavyDepsBanner() {
  const { detail, clear } = useHeavyDeps();
  if (!detail) return null;

  return (
    <div className="flex items-start justify-between gap-3 border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/60 dark:text-amber-200">
      <div>
        <span className="font-semibold">Model dependencies unavailable on the backend: </span>
        {detail}
      </div>
      <button
        type="button"
        onClick={clear}
        className="shrink-0 text-amber-700 underline hover:text-amber-900 dark:text-amber-300 dark:hover:text-amber-100"
      >
        dismiss
      </button>
    </div>
  );
}
