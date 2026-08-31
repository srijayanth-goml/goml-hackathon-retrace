export type TabKey = "compare" | "erasure" | "revisions" | "reports";

const TABS: { key: TabKey; label: string }[] = [
  { key: "compare", label: "Compare & Chat" },
  { key: "erasure", label: "Submit Erasure Request" },
  { key: "revisions", label: "Revisions" },
  { key: "reports", label: "Reports" },
];

interface TabNavProps {
  active: TabKey;
  onChange: (tab: TabKey) => void;
}

export function TabNav({ active, onChange }: TabNavProps) {
  return (
    <nav className="flex gap-1 border-b border-slate-200 px-4 dark:border-slate-700">
      {TABS.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onChange(tab.key)}
          className={
            "rounded-t-md px-3 py-2 text-sm font-medium transition " +
            (active === tab.key
              ? "border-b-2 border-slate-900 text-slate-900 dark:border-white dark:text-white"
              : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200")
          }
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
