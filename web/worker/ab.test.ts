import { afterEach, describe, expect, it, vi } from "vitest";
import { draw, parseArm, readCookie } from "./ab";

describe("parseArm", () => {
  it("accepts known arms and rejects anything else", () => {
    expect(parseArm("a")).toBe("a");
    expect(parseArm("control")).toBe("control");
    expect(parseArm("z")).toBeUndefined();
    expect(parseArm(null)).toBeUndefined();
    expect(parseArm("__proto__")).toBeUndefined();
  });
});

describe("readCookie", () => {
  it("finds a cookie among several and keeps '=' inside values", () => {
    expect(readCookie("x=1; sema-ab=a; y=2", "sema-ab")).toBe("a");
    expect(readCookie("t=a=b", "t")).toBe("a=b");
    expect(readCookie(null, "sema-ab")).toBeUndefined();
    expect(readCookie("other=1", "sema-ab")).toBeUndefined();
  });
});

describe("draw", () => {
  afterEach(() => vi.restoreAllMocks());

  it("weights the roll over the active arms", () => {
    const weights = { control: 1, a: 1, b: 2 };
    vi.spyOn(Math, "random").mockReturnValue(0.1);
    expect(draw(weights)).toBe("control");
    vi.spyOn(Math, "random").mockReturnValue(0.3);
    expect(draw(weights)).toBe("a");
    vi.spyOn(Math, "random").mockReturnValue(0.9);
    expect(draw(weights)).toBe("b");
  });

  it("never returns a zero-weight arm", () => {
    const weights = { control: 0, a: 1, b: 0 };
    for (const roll of [0, 0.5, 0.999]) {
      vi.spyOn(Math, "random").mockReturnValue(roll);
      expect(draw(weights)).toBe("a");
    }
  });
});
