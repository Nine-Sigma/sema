import * as React from "react";

type Theme = "dark" | "light";
const KEY = "sema-theme";

function resolve(): Theme {
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped === "dark" || stamped === "light") return stamped;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/* Theme lives on html[data-theme]; unset means "follow the system". localStorage is a convenience only. */
function useTheme(): { theme: Theme; setTheme: (t: Theme) => void } {
  const [theme, set] = React.useState<Theme>("dark");
  React.useEffect(() => {
    try {
      const saved = localStorage.getItem(KEY);
      if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
    } catch {
      /* storage unavailable */
    }
    set(resolve());
  }, []);
  const setTheme = React.useCallback((t: Theme) => {
    document.documentElement.setAttribute("data-theme", t);
    try {
      localStorage.setItem(KEY, t);
    } catch {
      /* storage unavailable */
    }
    set(t);
  }, []);
  return { theme, setTheme };
}

export { useTheme, type Theme };
