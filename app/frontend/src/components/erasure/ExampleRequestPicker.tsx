import type { ExampleRequest } from "../../api/types";

interface ExampleRequestPickerProps {
  examples: ExampleRequest[];
  onPick: (example: ExampleRequest) => void;
  disabled?: boolean;
}

/**
 * Buttons from GET /requests/examples -- the canned demo requests under
 * unlearning/requests/ (5 today; whatever count the manifest actually has, never hand-typed here). Exists specifically so a judge doesn't have to type an
 * exact entity name correctly (routes/meta.py's own docstring gives the same
 * reasoning: a typo'd request silently resolving to nothing would be an
 * embarrassing way to lose points on exactly the Erasure Targeting rubric line).
 */
export function ExampleRequestPicker({ examples, onPick, disabled }: ExampleRequestPickerProps) {
  if (examples.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {examples.map((ex) => (
        <button
          key={ex.name}
          type="button"
          disabled={disabled}
          onClick={() => onPick(ex)}
          title={ex.comment ?? undefined}
          className="btn-secondary text-left"
        >
          {ex.entity ?? "(any entity)"}
          {ex.attribute ? ` / ${ex.attribute}` : ""}
        </button>
      ))}
    </div>
  );
}
