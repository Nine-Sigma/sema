import { type Bus, docTop } from "./bus";

/* Chapter state. Each chapter element carries data-chapter; the active chapter is the last one whose
   top has passed the middle of the viewport. A pinned stage may hand over to its landing chapter
   part-way through its travel: data-chapter-next + data-chapter-at (fraction of the travel).
   Writes html[data-chapter] and html[data-scrolled]; the plane and the nav read them. */
type Entry = { el: HTMLElement; name: string; top: number; next?: string; at?: number; travel: number };

export function trackChapters(bus: Bus): () => void {
  const root = document.documentElement;
  let entries: Entry[] = [];

  const measure = (): void => {
    entries = [...document.querySelectorAll<HTMLElement>("main [data-chapter]")].map((el) => ({
      el,
      name: el.dataset.chapter ?? "",
      top: docTop(el),
      next: el.dataset.chapterNext,
      at: el.dataset.chapterAt ? Number(el.dataset.chapterAt) : undefined,
      travel: Math.max(0, el.offsetHeight - window.innerHeight),
    }));
  };

  const offLayout = bus.onRelayout(measure);
  const offFrame = bus.subscribe(({ y, vh }) => {
    const line = y + vh * 0.5;
    let current = entries[0]?.name ?? "";
    for (const e of entries) {
      const pinned = e.next !== undefined && e.at !== undefined;
      if (pinned ? y < e.top : line < e.top) break;
      current = e.name;
      if (pinned && e.next && e.at !== undefined && y >= e.top + e.travel * e.at) current = e.next;
    }
    if (root.dataset.chapter !== current) root.dataset.chapter = current;
    const scrolled = y > 8;
    if (scrolled !== root.hasAttribute("data-scrolled")) root.toggleAttribute("data-scrolled", scrolled);
  });
  return () => {
    offLayout();
    offFrame();
  };
}

/* Anchors into a pinned stage land at the end of its travel (the resolved state), not at its top. */
export function routeStageAnchors(): () => void {
  const target = (hash: string): { top: number; behavior: ScrollBehavior } | null => {
    if (!hash.startsWith("#") || hash.length < 2) return null;
    const el = document.getElementById(decodeURIComponent(hash.slice(1)));
    const stage = el?.closest<HTMLElement>(".sema-stage");
    if (!el || !stage || !document.documentElement.hasAttribute("data-motion")) return null;
    const travel = Math.max(0, stage.offsetHeight - window.innerHeight);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    return { top: docTop(stage) + travel, behavior: reduce ? "auto" : "smooth" };
  };
  const onClick = (ev: MouseEvent): void => {
    const a = (ev.target as Element | null)?.closest<HTMLAnchorElement>('a[href^="#"]');
    if (!a || ev.defaultPrevented) return;
    const t = target(a.getAttribute("href") ?? "");
    if (!t) return;
    ev.preventDefault();
    window.scrollTo(t);
    history.replaceState(null, "", a.getAttribute("href"));
  };
  document.addEventListener("click", onClick);
  const t = target(location.hash);
  if (t) requestAnimationFrame(() => window.scrollTo({ top: t.top, behavior: "auto" }));
  return () => document.removeEventListener("click", onClick);
}
