import { DemoShell } from "../../components/demo-shell";

export default function DemoPage() {
  return (
    <div className="page-shell py-12 md:py-16">
      <div className="mb-8 max-w-3xl">
        <span className="eyebrow">Phase 1 Demo</span>
        <h1 className="mt-5 text-4xl font-semibold tracking-tight">Product shell for DermAI</h1>
        <p className="mt-4 text-base leading-7 text-[var(--muted)]">
          This is the first full-stack interface pass. The shell is ready for retrieval, streaming chat,
          classifier inference, and combined multimodal sessions in later phases.
        </p>
      </div>

      <DemoShell />
    </div>
  );
}
