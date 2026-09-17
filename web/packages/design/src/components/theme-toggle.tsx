import * as React from "react";
import { Toggle } from "./ui/toggle";
import { useTheme } from "../hooks/use-theme";

/* Half-disc icon; pressed = light, and the disc rotates 180°. */
function ThemeToggle(props: Omit<React.ComponentProps<typeof Toggle>, "pressed" | "onPressedChange">) {
  const { theme, setTheme } = useTheme();
  const light = theme === "light";
  return (
    <Toggle
      pressed={light}
      onPressedChange={(on) => setTheme(on ? "light" : "dark")}
      aria-label={light ? "Switch to dark theme" : "Switch to light theme"}
      {...props}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.25" />
        <path d="M8 1.75A6.25 6.25 0 0 1 8 14.25Z" fill="currentColor" />
      </svg>
    </Toggle>
  );
}

export { ThemeToggle };
