const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

async function animateLearningPath() {
  if (prefersReducedMotion) return;
  const nodes = document.querySelectorAll("[data-learning-node]");
  if (!nodes.length) return;
  try {
    const { gsap } = await import("https://cdn.jsdelivr.net/npm/gsap@3.12.5/index.js");
    gsap.from(nodes, {
      autoAlpha: 0,
      y: 16,
      duration: 0.38,
      ease: "power2.out",
      stagger: 0.06,
    });
  } catch (error) {
    console.info("[ISYARA UI] GSAP unavailable; rendering without motion.", error);
  }
}

animateLearningPath();
