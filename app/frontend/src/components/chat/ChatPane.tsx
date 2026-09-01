import { forwardRef, useImperativeHandle, useState } from "react";
import { postChat } from "../../api/client";
import { useHeavyDeps } from "../layout/HeavyDepsContext";
import type { ChatMessage } from "../../api/types";

export interface ChatPaneHandle {
  /** Sends `text` as a user turn against this pane's revision. Exposed via ref so
   * CompareChat's "send to both" input can drive two independently-stateful panes
   * from one action -- each pane still owns its own transcript. */
  sendMessage: (text: string) => void;
}

interface ChatPaneProps {
  revision: number;
  title: string;
}

/**
 * One revision's conversation: message list + input, POST /chat
 * {revision, messages, max_new_tokens?}. Stateless per Module 5's design -- the
 * full transcript lives only in this component's own state (so this pane surviving
 * a backend restart mid-demo is really "the browser tab kept the history", worth
 * knowing if a judge asks why refreshing the page clears the chat) -- but only the
 * CURRENT question is sent to /chat, never the prior turns. Every finetuning/
 * unlearning training record is a single isolated {user, assistant} pair (see
 * data_pipeline/augment/*.py) -- no multi-turn example exists anywhere in the
 * training data -- so resending accumulated history the way a normal chat client
 * would put the model in a conversation shape it never saw a single example of,
 * which measurably degraded answer quality (see the RCA on live-chat vs.
 * finetuning/eval_quick.py's accuracy). Each turn is scored independently by the
 * model, same as finetuning/eval_quick.py's own single-turn generate_answer.
 */
export const ChatPane = forwardRef<ChatPaneHandle, ChatPaneProps>(function ChatPane({ revision, title }, ref) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { reportIfHeavyDepsMissing } = useHeavyDeps();

  const send = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || sending) return;

    const userTurn: ChatMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userTurn]);
    setInput("");
    setSending(true);
    setError(null);

    // Only this one turn goes to the model -- see the component docstring above for
    // why: the adapter was never trained on multi-turn context, so sending the
    // accumulated transcript (the "obvious" chat-client behavior) actively hurts
    // answer quality rather than helping it.
    postChat({ revision, messages: [userTurn] })
      .then((res) => {
        setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
      })
      .catch((err: unknown) => {
        reportIfHeavyDepsMissing(err);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setSending(false));
  };

  useImperativeHandle(ref, () => ({ sendMessage: send }));

  return (
    <div className="card flex h-[28rem] flex-col">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
        {sending && <span className="text-xs text-slate-400">generating...</span>}
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto rounded-md bg-slate-50 p-2 dark:bg-slate-950/40">
        {messages.length === 0 && (
          <p className="text-sm italic text-slate-400">No messages yet -- ask this revision something.</p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={
              "max-w-[85%] rounded-lg px-3 py-1.5 text-sm " +
              (m.role === "user"
                ? "ml-auto bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "bg-white text-slate-800 shadow-sm dark:bg-slate-800 dark:text-slate-100")
            }
          >
            {m.content}
          </div>
        ))}
      </div>

      {error && <p className="mt-1 text-xs text-rose-600 dark:text-rose-400">{error}</p>}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="mt-2 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask this revision..."
          disabled={sending}
          className="flex-1 rounded-md border border-slate-300 px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
        />
        <button type="submit" disabled={sending || !input.trim()} className="btn">
          Send
        </button>
      </form>
    </div>
  );
});
