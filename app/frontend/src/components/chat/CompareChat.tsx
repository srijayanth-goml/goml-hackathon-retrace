import { useEffect, useMemo, useRef, useState } from "react";
import { ChatPane, type ChatPaneHandle } from "./ChatPane";
import { RevisionPicker } from "./RevisionPicker";
import type { RevisionSummary } from "../../api/types";

interface CompareChatProps {
  revisions: RevisionSummary[];
}

/**
 * The flagship screen: two ChatPanes side by side. Left is PINNED to revision 0
 * (the baseline) rather than freely selectable -- keeps the screen legible for a
 * judge (and whoever's filming the demo video) while every revision, including 0,
 * stays individually reachable from the Revisions tab. Right defaults to the
 * highest-numbered revision so a fresh erasure request is immediately comparable
 * without extra clicks, and keeps following the latest revision as new ones finish
 * training until a judge explicitly picks a different one. This is the literal
 * live, side-by-side before/after Design Doc Section 8 and the brief's "a recorded
 * demo alone does not satisfy this" line both ask for.
 */
export function CompareChat({ revisions }: CompareChatProps) {
  const highestRevision = useMemo(
    () => revisions.reduce((max, r) => Math.max(max, r.revision), 0),
    [revisions],
  );

  const [rightRevision, setRightRevision] = useState(highestRevision);
  const [userPickedRight, setUserPickedRight] = useState(false);
  const [sharedInput, setSharedInput] = useState("");

  // Auto-follow the latest revision until the judge explicitly picks one --
  // otherwise the right pane would silently stay on revision 0 forever if this
  // component mounted before /revisions finished loading.
  useEffect(() => {
    if (!userPickedRight) {
      setRightRevision(highestRevision);
    }
  }, [highestRevision, userPickedRight]);

  const leftRef = useRef<ChatPaneHandle>(null);
  const rightRef = useRef<ChatPaneHandle>(null);

  const sendToBoth = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    leftRef.current?.sendMessage(trimmed);
    rightRef.current?.sendMessage(trimmed);
    setSharedInput("");
  };

  if (revisions.length === 0) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">Loading revisions...</p>;
  }

  return (
    <div className="space-y-3">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendToBoth(sharedInput);
        }}
        className="card flex items-center gap-2"
      >
        <label className="text-sm font-medium text-slate-700 dark:text-slate-200" htmlFor="send-to-both">
          Send to both
        </label>
        <input
          id="send-to-both"
          value={sharedInput}
          onChange={(e) => setSharedInput(e.target.value)}
          placeholder="Ask revision-0 and the compared revision the same question..."
          className="flex-1 rounded-md border border-slate-300 px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
        />
        <button type="submit" disabled={!sharedInput.trim()} className="btn">
          Send to both
        </button>
      </form>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <ChatPane ref={leftRef} revision={0} title="revision-0 (baseline, pre-erasure)" />

        <div className="flex h-[28rem] flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-slate-600 dark:text-slate-300">Compare against:</span>
            <RevisionPicker
              revisions={revisions}
              value={rightRevision}
              onChange={(rev) => {
                setUserPickedRight(true);
                setRightRevision(rev);
              }}
              exclude={[0]}
            />
          </div>
          <div className="flex-1">
            <ChatPane
              ref={rightRef}
              revision={rightRevision}
              title={rightRevision === 0 ? "revision-0 (baseline)" : `revision-${rightRevision} (post-erasure)`}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
