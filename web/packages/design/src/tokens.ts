/* Typed mirror of theme.css. Use for canvases, charts, and anything that cannot read CSS variables.
   theme.css is the source of truth; keep these in step. */
export const colors = {
  dark: {
    ground: "#0a0d19",
    ink: "#e8e6df",
    muted: "#8a8f9c",
    rule: "#2a3040",
    ruleStrong: "#5c6478",
    accent: "#e5a23a",
    accentFill: "#e5a23a",
    onAccent: "#0a0d19",
    blue: "#8fb0d6",
  },
  light: {
    ground: "#f2f1ec",
    ink: "#0a0d19",
    muted: "#5c6270",
    rule: "#c9cbd2",
    ruleStrong: "#7f8490",
    accent: "#8f5e12",
    accentFill: "#b87a1e",
    onAccent: "#0a0d19",
    blue: "#3e6a99",
  },
} as const;

export const fonts = {
  sans: '"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif',
  mono: '"IBM Plex Mono", "SFMono-Regular", Menlo, monospace',
  serif: '"Cormorant Garamond", Georgia, serif',
} as const;

/* Google Fonts request that matches the weights the design uses. */
export const fontsHref =
  "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500;700&family=Cormorant+Garamond:wght@500&display=swap";

export const motion = {
  ease: "cubic-bezier(0.16, 1, 0.3, 1)",
  durations: { press: 120, hover: 180, toggle: 220, state: 260, ring: 320, arrive: 700, trace: 500, pulse: 6000 },
  /* Hero moment, ms from release: nodes 0+8/rank, edges 200+6/i, amber 500+40/i, rings 640+40/i, ticks 760, core 820, pulse loop from 900. */
  heroFallbackMs: 4000,
} as const;

export const breakpoints = { sm: 561, lg: 901 } as const;

export type ThemeName = keyof typeof colors;
export type ColorToken = keyof typeof colors.dark;
