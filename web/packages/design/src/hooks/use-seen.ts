import * as React from "react";

/* Sets data-seen on the element the first time 60% of it is on screen. Without IO, seen at once. */
function useSeen<T extends HTMLElement>(threshold = 0.6): React.RefObject<T | null> {
  const ref = React.useRef<T>(null);
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!("IntersectionObserver" in window)) {
      el.dataset.seen = "";
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            el.dataset.seen = "";
            io.unobserve(el);
          }
        }
      },
      { threshold },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold]);
  return ref;
}

export { useSeen };
