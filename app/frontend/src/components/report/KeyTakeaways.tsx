export function KeyTakeaways({ takeaways }: { takeaways: string[] }) {
  return (
    <section className="card space-y-2">
      <h3 className="font-semibold text-slate-900 dark:text-slate-100">Key Takeaways</h3>
      {takeaways.length === 0 ? (
        <p className="text-sm text-slate-500 dark:text-slate-400">No takeaways generated.</p>
      ) : (
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800 dark:text-slate-100">
          {takeaways.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
