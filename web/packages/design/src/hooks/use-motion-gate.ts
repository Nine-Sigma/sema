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
    const fallback = window.setTimeout(play, 4000);
    return () => {
      reduce.removeEventListener("change", onChange);
      window.removeEventListener("load", go);
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
