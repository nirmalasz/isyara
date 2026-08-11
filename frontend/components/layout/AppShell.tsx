import Link from "next/link";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-isyara-background">
      <header className="sticky top-0 z-30 border-b border-isyara-tint bg-white/90 backdrop-blur">
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <Link href="/" className="font-extrabold text-isyara-dark">
            ISYARA
          </Link>
          <div className="flex gap-2 text-sm font-semibold">
            <Link className="rounded px-3 py-2 text-isyara-primary hover:bg-isyara-tint" href="/">
              Translator
            </Link>
            <Link className="rounded px-3 py-2 text-slate-700 hover:bg-isyara-tint" href="/history">
              Riwayat
            </Link>
            <Link className="rounded px-3 py-2 text-slate-700 hover:bg-isyara-tint" href="/profile">
              Profil
            </Link>
          </div>
        </nav>
      </header>
      <main>{children}</main>
    </div>
  );
}
