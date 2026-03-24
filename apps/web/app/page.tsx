import Link from "next/link";

const pillars = [
  {
    title: "Evidence-grounded dermatology chat",
    body: "DermAI is being rebuilt around retrieval-backed answers, visible sources, and cautious medical UX.",
  },
  {
    title: "Lesion analysis workflow",
    body: "The image pipeline will expose classifier predictions, confidence handling, and Grad-CAM overlays in a single session.",
  },
  {
    title: "Portfolio-grade product shell",
    body: "The interface is designed as a shareable web product rather than a research notebook or internal prototype.",
  },
];

export default function HomePage() {
  return (
    <div className="py-14 md:py-20">
      <section className="page-shell">
        <div className="glass-card overflow-hidden rounded-[2.5rem] px-6 py-10 md:px-10 md:py-14">
          <div className="grid gap-10 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
            <div>
              <span className="eyebrow">Dermatology AI, rebuilt</span>
              <h1 className="mt-6 max-w-3xl text-4xl font-semibold leading-tight tracking-tight md:text-6xl">
                A modern dermatology assistant with grounded answers and lesion-aware workflows.
              </h1>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-[var(--muted)]">
                DermAI turns the original conference direction into a product-grade web experience:
                dermatology chat, evidence tracing, image analysis, and confidence-aware guidance in one
                system.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  href="/demo"
                  className="rounded-full bg-[var(--accent)] px-6 py-3 text-sm font-semibold text-white"
                >
                  Open Demo Shell
                </Link>
                <Link
                  href="/about"
                  className="rounded-full border border-black/10 bg-white px-6 py-3 text-sm font-semibold"
                >
                  Read The Build Direction
                </Link>
              </div>
            </div>

            <div className="grid gap-4">
              <div className="rounded-[2rem] border border-black/10 bg-white/80 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  Product arc
                </p>
                <p className="mt-3 text-2xl font-semibold leading-tight">
                  Chat, retrieval, vision, then multimodal reasoning.
                </p>
              </div>
              <div className="rounded-[2rem] border border-black/10 bg-[var(--accent)] px-5 py-6 text-white">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-white/70">
                  Build stance
                </p>
                <p className="mt-3 text-lg leading-7">
                  Grounded first. Confidence-aware by default. Replaceable model providers. Product shell
                  before infrastructure sprawl.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="page-shell mt-8 grid gap-4 md:grid-cols-3">
        {pillars.map((pillar) => (
          <div key={pillar.title} className="glass-card rounded-[2rem] p-6">
            <p className="text-lg font-semibold">{pillar.title}</p>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{pillar.body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
