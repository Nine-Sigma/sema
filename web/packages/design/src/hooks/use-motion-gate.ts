import * as React from "react";

/* Owns html[data-motion]. Reduced motion: attribute never set, end state renders.
   Otherwise "pending" holds the hero at its first frame until load + fonts + two painted frames,
   so a hidden or still-loading iframe does not burn the moment before the reader sees it. */
function useMotionGate(): { replay: () => void } {
  const play = React.useCallback(() => {
    const root = document.documentElement;
    if (root.getAttribute("data-motion") !== "pending") return;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (root.getAttribute("data-motion") === "pending") root.setAttribute("data-motion", "");
      });
    });
  }, []);

  React.useEffect(() => {
    const root = document.documentElement;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reduce.matches) return;
    root.setAttribute("data-motion", "pending");
    const onChange = (): void => {
      if (reduce.matches) root.removeAttribute("data-motion");
    };
    reduce.addEventListener("change", onChange);
    const fonts = document.fonts?.ready ?? Promise.resolve();
    const go = (): void => {
      void fonts.then(play, play);
    };
    if (document.readyState === "complete") go();
    else window.addEventListener("load", go, { once: true });
    /* Fallback without rAF: a headless or throttled renderer delivers no frames, so after 4s a visible
       document releases directly. A hidden tab keeps the moment until it is shown. */
    const release = (): void => {
      if (root.getAttribute("data-motion") === "pending") root.setAttribute("data-motion", "");
    };
    const onVisible = (): void => {
      if (document.visibilityState === "visible") play();
    };
    const fallback = window.setTimeout(() => {
      if (document.visibilityState === "hidden") document.addEventListener("visibilitychange", onVisible);
      else release();
    }, 4000);
    return () => {
      reduce.removeEventListener("change", onChange);
      window.removeEventListener("load", go);
      document.removeEventListener("visibilitychange", onVisible);
      window.clearTimeout(fallback);
    };
  }, [play]);

  const replay = React.useCallback(() => {
    const root = document.documentElement;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    root.removeAttribute("data-motion");
    void root.offsetWidth;
    root.setAttribute("data-motion", "pending");
    play();
  }, [play]);

  return { replay };
}

export { useMotionGate };
