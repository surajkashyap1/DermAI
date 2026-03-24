import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "DermAI",
  description:
    "DermAI is a dermatology assistant with grounded chat and lesion image support.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <header className="sticky top-0 z-20 border-b border-white/8 bg-[rgba(6,14,19,0.72)] backdrop-blur-xl">
          <div className="page-shell flex items-center justify-between py-4">
            <Link href="/" className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,var(--accent),#158f93)] text-sm font-bold text-[#021012] shadow-[0_12px_30px_rgba(34,193,161,0.28)]">
                D
              </div>
              <div className="text-lg font-semibold tracking-tight">DermAI</div>
            </Link>
          </div>
        </header>

        <main>{children}</main>
      </body>
    </html>
  );
}
