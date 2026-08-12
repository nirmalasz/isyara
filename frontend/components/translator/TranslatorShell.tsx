"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";

export function TranslatorShell() {
  const pageRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !pageRef.current) return;
    gsap.from(pageRef.current.querySelectorAll("[data-panel]"), {
      autoAlpha: 0,
      y: 14,
      duration: 0.4,
      ease: "power2.out",
      stagger: 0.08,
    });
  }, []);

  return (
    <section ref={pageRef} className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div data-panel className="rounded-card border border-isyara-tint bg-white p-6 shadow-sm sm:p-8">
        <p className="text-sm font-semibold uppercase tracking-[0.16em] text-isyara-primary">ISYARA Translator</p>
        <h1 className="mt-3 text-3xl font-semibold text-isyara-dark sm:text-4xl">BISINDO ke suara, ucapan ke teks.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">Next.js shell ini mengikuti design system ISYARA dan memakai API Django/FastAPI translator yang sama.</p>
      </div>
      <div data-panel className="mt-6 grid gap-5 lg:grid-cols-[1fr_360px]">
        <div className="rounded-card border border-slate-200 bg-slate-950 p-5 text-white shadow-sm">
          <div className="grid aspect-video place-items-center rounded-card border border-white/10">
            <p className="text-sm font-medium text-slate-300">Camera + MediaPipe overlay lives in the Django translator page.</p>
          </div>
        </div>
        <aside className="rounded-card border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold text-isyara-dark">Status AI</h2>
          <p className="mt-3 rounded-card bg-isyara-background p-4 text-sm font-medium text-slate-700">Model penerjemah sedang disiapkan. Prediksi nyata aktif setelah `sign_classifier.pt` tersedia.</p>
        </aside>
      </div>
    </section>
  );
}
