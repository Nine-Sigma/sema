import * as React from "react";
import { createBus, type Bus } from "./bus";
import type { PlaneHandle } from "./plane-state";

/* The motion clock and the plane handle, shared by the page. The plane registers its handle on
   mount (it renders before <main>, so its effect runs before the stages read the ref). */
export type Motion = { bus: Bus; plane: React.MutableRefObject<PlaneHandle | null> };

const MotionContext = React.createContext<Motion | null>(null);

export function MotionProvider({ children }: { children: React.ReactNode }) {
  const [motion] = React.useState<Motion>(() => ({ bus: createBus(), plane: { current: null } }));
  React.useEffect(() => {
    const detach = motion.bus.attach();
    return () => {
      detach();
      motion.bus.destroy();
    };
  }, [motion]);
  return <MotionContext.Provider value={motion}>{children}</MotionContext.Provider>;
}

export function useMotion(): Motion {
  const m = React.useContext(MotionContext);
  if (!m) throw new Error("useMotion outside MotionProvider");
  return m;
}
