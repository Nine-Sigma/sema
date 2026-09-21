import * as React from "react";
import { clamp, docTop, motionOn, type Frame } from "./bus";
import { useMotion } from "./context";

/* Progress of a pinned stage: 0 when its top reaches the viewport top, 1 when its travel
   (height − viewport) is spent. The callback runs only when the value changes. */
export function useStage<T extends HTMLElement>(onFrame: (p: number, f: Frame) => void): React.RefObject<T | null> {
  const ref = React.useRef<T>(null);
  const cb = React.useRef(onFrame);
  cb.current = onFrame;
  const { bus } = useMotion();
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let top = 0;
    let travel = 0;
    let last = -1;
    const offLayout = bus.onRelayout(() => {
      top = docTop(el);
      travel = Math.max(0, el.offsetHeight - window.innerHeight);
      last = -1;
    });
    const off = bus.subscribe((f) => {
      if (!motionOn()) return;
      const p = travel <= 0 ? (f.y >= top ? 1 : 0) : clamp((f.y - top) / travel, 0, 1);
      if (p === last) return;
      last = p;
      cb.current(p, f);
    });
    bus.relayout();
    return () => {
      offLayout();
      off();
    };
  }, [bus]);
  return ref;
}

/* Sets --reveal (0 → 1) on every `selector` inside the container as it rises `distance` px into
   the viewport from the bottom edge. The chapter numerals draw with it. */
export function useReveal<T extends HTMLElement>(selector: string, distance = 200): React.RefObject<T | null> {
  const ref = React.useRef<T>(null);
  const { bus } = useMotion();
  React.useEffect(() => {
    const host = ref.current;
    if (!host) return;
    const items = [...host.querySelectorAll<HTMLElement>(selector)];
    const last = new Map<HTMLElement, number>();
    const off = bus.subscribe((f) => {
      if (!motionOn()) return;
      for (const el of items) {
        const r = el.getBoundingClientRect();
        const v = clamp((f.vh - r.top) / distance, 0, 1);
        if (last.get(el) === v) continue;
        last.set(el, v);
        el.style.setProperty("--reveal", v.toFixed(3));
      }
    });
    return off;
  }, [bus, selector, distance]);
  return ref;
}
