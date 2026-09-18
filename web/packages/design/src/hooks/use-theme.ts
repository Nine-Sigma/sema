import * as React from "react";

type Theme = "dark" | "light";
type ThemeSource = "system" | "pinned";
const KEY = "sema-theme";
const LIGHT = "(prefers-color-scheme: light)";

function stamped(): Theme | null {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "dark" || t === "light" ? t : null;
}

function resolve(): Theme {
  return stamped() ?? (window.matchMedia(LIGHT).matches ? "light" : "dark");
}

/* One store for every instance: html[data-theme] is the state, the observer is the subscription. */
function subscribe(onChange: () => void): () => void {
  const mo = new MutationObserver(onChange);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const mq = window.matchMedia(LIGHT);
  mq.addEventListener("change", onChange);
  return () => {
    mo.disconnect();
    mq.removeEventListener("change", onChange);
  };
}

function snapshot(): string {
  return `${resolve()}:${stamped() ? "pinned" : "system"}`;
}

function remember(t: Theme | null): void {
  try {
    if (t) localStorage.setItem(KEY, t);
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable */
  }
}

/* Theme lives on html[data-theme]; unset means "follow the system". localStorage is a convenience only;
   apps apply it before first paint with public/theme-boot.js (same KEY). setTheme("system") unpins. */
function useTheme(): { theme: Theme; source: ThemeSource; setTheme: (t: Theme | "system") => void } {
  const snap = React.useSyncExternalStore(subscribe, snapshot, () => "dark:system");
  const [theme, source] = snap.split(":") as [Theme, ThemeSource];
  React.useEffect(() => {
    if (stamped()) return;
    try {
      const saved = localStorage.getItem(KEY);
      if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
    } catch {
      /* storage unavailable */
    }
  }, []);
  const setTheme = React.useCallback((t: Theme | "system") => {
    if (t === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    remember(t === "system" ? null : t);
  }, []);
  return { theme, source, setTheme };
}

export { useTheme, type Theme, type ThemeSource };
