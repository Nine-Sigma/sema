/* One clock for every scroll-driven value on the page. Scroll and resize mark a frame dirty; one
   requestAnimationFrame runs the subscribers in order with the same reading of the scroll position,
   so the plane, the stages and the numerals never disagree by a frame. A subscriber that returns
   true asks for another frame (a lerp still settling). */

export type Frame = { y: number; vw: number; vh: number };
export type Subscriber = (f: Frame) => boolean | void;

export type Bus = {
  /** post: run after the ordinary subscribers in the same frame (the plane composes last). */
  subscribe: (fn: Subscriber, post?: boolean) => () => void;
  /** Ask for a frame now (after a layout change or a pointer move). */
  request: () => void;
  /** Recompute cached document offsets on the next frame. */
  relayout: () => void;
  onRelayout: (fn: () => void) => () => void;
  destroy: () => void;
};

export function createBus(): Bus {
  const subs = new Set<Subscriber>();
  const postSubs = new Set<Subscriber>();
  const layoutSubs = new Set<() => void>();
  let queued = false;
  let needsLayout = true;

  const frame = (): void => {
    queued = false;
    if (needsLayout) {
      needsLayout = false;
      layoutSubs.forEach((fn) => fn());
    }
    const f: Frame = { y: window.scrollY, vw: window.innerWidth, vh: window.innerHeight };
    let again = false;
    subs.forEach((fn) => {
      if (fn(f) === true) again = true;
    });
    postSubs.forEach((fn) => {
      if (fn(f) === true) again = true;
    });
    if (again) request();
  };
  const request = (): void => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(frame);
  };
  const relayout = (): void => {
    needsLayout = true;
    request();
  };
  window.addEventListener("scroll", request, { passive: true });
  window.addEventListener("resize", relayout);
  const ro = "ResizeObserver" in window ? new ResizeObserver(relayout) : null;
  ro?.observe(document.body);
  const fonts = document.fonts?.ready;
  void fonts?.then(relayout, () => undefined);

  return {
    subscribe: (fn, post = false) => {
      const set = post ? postSubs : subs;
      set.add(fn);
      request();
      return () => set.delete(fn);
    },
    request,
    relayout,
    onRelayout: (fn) => {
      layoutSubs.add(fn);
      return () => layoutSubs.delete(fn);
    },
    destroy: () => {
      window.removeEventListener("scroll", request);
      window.removeEventListener("resize", relayout);
      ro?.disconnect();
      subs.clear();
      postSubs.clear();
      layoutSubs.clear();
    },
  };
}

export const clamp = (v: number, a: number, b: number): number => Math.min(b, Math.max(a, v));
/** Progress of p between a and b, clamped to [0, 1]. */
export const seg = (p: number, a: number, b: number): number => clamp((p - a) / (b - a), 0, 1);
/** Cubic ease-out: the --ease-signal curve on a scrubbed timeline. */
export const ease = (t: number): number => 1 - Math.pow(1 - t, 3);
/** Top of an element in document coordinates. */
export const docTop = (el: Element): number => el.getBoundingClientRect().top + window.scrollY;
export const motionOn = (): boolean => document.documentElement.hasAttribute("data-motion");
