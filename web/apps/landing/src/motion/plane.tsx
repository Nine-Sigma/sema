import * as React from "react";
import { useMotion } from "./context";
import { PlaneController } from "./plane-state";

/* The graph plane: one fixed layer behind the page. Owns nothing of the copy; the stages drive it
   through the handle, the chapter tracker through html[data-chapter]. Static grain on top. */
export function GraphPlane() {
  const { bus, plane } = useMotion();
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const host = ref.current;
    if (!host) return;
    const ctl = new PlaneController(host);
    plane.current = ctl;
    const root = document.documentElement;
    ctl.setChapter(root.dataset.chapter ?? "hero");
    const offLayout = bus.onRelayout(() => ctl.relayout());
    const offFrame = bus.subscribe((f) => ctl.frame(f), true);
    const mo = new MutationObserver(() => {
      ctl.setChapter(root.dataset.chapter ?? "hero");
      bus.request();
    });
    mo.observe(root, { attributes: true, attributeFilter: ["data-chapter"] });
    const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    const onPointer = (e: PointerEvent): void => {
      ctl.pointerAt((e.clientX / window.innerWidth) * 2 - 1, (e.clientY / window.innerHeight) * 2 - 1);
      bus.request();
    };
    if (fine) window.addEventListener("pointermove", onPointer, { passive: true });
    bus.relayout();
    return () => {
      offLayout();
      offFrame();
      mo.disconnect();
      window.removeEventListener("pointermove", onPointer);
      plane.current = null;
      ctl.destroy();
    };
  }, [bus, plane]);

  return (
    <div ref={ref} className="sema-plane" aria-hidden="true">
      <div className="sema-grain" />
    </div>
  );
}
