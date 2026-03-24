import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "DermAI",
  description:
    "DermAI is a dermatology evidence and lesion-analysis demo built as a modern full-stack web product.",
};

const navItems = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Demo" },
  { href: "/about", label: "About" },
  { href: "/privacy-disclaimer", label: "Disclaimer" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <header className="sticky top-0 z-20 border-b border-black/5 bg-white/70 backdrop-blur-xl">
          <div className="page-shell flex items-center justify-between py-4">
            <Link href="/" className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[var(--accent)] text-sm font-bold text-white">
                D
              </div>
              <div>
                <div className="text-lg font-semibold tracking-tight">DermAI</div>
                <div className="text-xs text-[var(--muted)]">
                  Dermatology evidence and lesion intelligence
                </div>
              </div>
            </Link>

            <nav className="hidden items-center gap-6 text-sm text-[var(--muted)] md:flex">
              {navItems.map((item) => (
                <Link key={item.href} href={item.href} className="transition hover:text-[var(--foreground)]">
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>

        <main>{children}</main>

        <footer className="border-t border-black/5 bg-white/60 py-8">
          <div className="page-shell flex flex-col gap-3 text-sm text-[var(--muted)] md:flex-row md:items-center md:justify-between">
            <p>DermAI Phase 1 foundation build.</p>
            <p>Portfolio product rebuild aligned with the conference paper direction.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
