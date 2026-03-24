const sections = [
  {
    title: "Why DermAI exists",
    body: "The project started from a conference paper direction that combined dermatology retrieval and lesion classification. The rebuild turns that into a product-grade website that can actually be shared and demoed.",
  },
  {
    title: "What the rebuild preserves",
    body: "Dermatology scope, grounded answers, confidence-aware behavior, lesion analysis, and explainability remain core to the identity of the project.",
  },
  {
    title: "What changes in the rebuild",
    body: "The implementation moves toward a modern stack: Next.js, FastAPI, modular services, hybrid retrieval, graph workflows, and explicit evaluation hooks.",
  },
];

export default function AboutPage() {
  return (
    <div className="page-shell py-12 md:py-16">
      <span className="eyebrow">About DermAI</span>
      <h1 className="mt-5 max-w-3xl text-4xl font-semibold tracking-tight">Paper-aligned, product-first.</h1>
      <p className="mt-4 max-w-3xl text-base leading-8 text-[var(--muted)]">
        DermAI is no longer being treated as a one-off conference artifact. The rebuild aims for a modern
        portfolio product with clean interfaces, cautious medical UX, and infrastructure that can evolve
        phase by phase.
      </p>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        {sections.map((section) => (
          <div key={section.title} className="glass-card rounded-[2rem] p-6">
            <h2 className="text-lg font-semibold">{section.title}</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{section.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
