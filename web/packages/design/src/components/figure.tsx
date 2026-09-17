import * as React from "react";
import { cn } from "../lib/utils";

export type FigureSource = { id: string; title: string; desc: string; svg: string };

type FigureProps = React.ComponentProps<"div"> & {
  source: FigureSource;
  /** wide: scrolls sideways under 720px content width. plate: 1px rule around it. */
  variant?: "default" | "wide" | "plate";
  /** Figma plates are drawn for 640-1200px; a mobile twin is swapped in under 900px. */
  mobile?: FigureSource;
};

/* Inline SVG from Figma, themed through the tokens at build time (scripts/build-figures.mjs).
   The markup is repository content, not user input. */
function Figure({ source, mobile, variant = "default", className, ...props }: FigureProps) {
  return (
    <>
      <div
        data-slot="figure"
        className={cn(
          "relative w-full [&_svg]:block [&_svg]:h-auto [&_svg]:w-full",
          variant === "wide" && "min-w-0 overflow-x-auto overscroll-x-contain [scrollbar-width:thin] [&_svg]:min-w-[720px]",
          variant === "plate" && "border border-rule",
          mobile && "max-lg:hidden",
          className,
        )}
        dangerouslySetInnerHTML={{ __html: source.svg }}
        {...props}
      />
      {mobile ? (
        <div
          data-slot="figure-mobile"
          className={cn("relative w-full max-w-[358px] lg:hidden [&_svg]:block [&_svg]:h-auto [&_svg]:w-full", className)}
          dangerouslySetInnerHTML={{ __html: mobile.svg }}
        />
      ) : null}
    </>
  );
}

export { Figure, type FigureProps };
