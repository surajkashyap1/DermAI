const disclaimers = [
  "DermAI is a software demo and does not replace a licensed clinician.",
  "Outputs should be treated as informational, not diagnostic or treatment advice.",
  "Low-confidence or emergency-pattern cases should escalate to direct medical care.",
  "Uploaded images and session content should be handled as sensitive health-adjacent data in later deployment phases.",
];

export default function PrivacyDisclaimerPage() {
  return (
    <div className="page-shell py-12 md:py-16">
      <span className="eyebrow">Privacy And Disclaimer</span>
      <h1 className="mt-5 text-4xl font-semibold tracking-tight">Cautious by default.</h1>
      <p className="mt-4 max-w-3xl text-base leading-8 text-[var(--muted)]">
        The product direction for DermAI assumes explicit safety messaging, transparent evidence use, and
        conservative handling of low-confidence outputs. This page is the initial placeholder for those
        policies.
      </p>

      <div className="mt-10 space-y-4">
        {disclaimers.map((item) => (
          <div key={item} className="glass-card rounded-[1.75rem] p-5 text-sm leading-7 text-[var(--foreground)]">
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}
