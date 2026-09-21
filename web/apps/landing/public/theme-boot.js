/* Runs before the stylesheet: applies the saved theme so a pinned theme never flashes, and holds the
   motion gate (html[data-motion="pending"]) so the prerendered end state does not paint before the
   load moment. @sema/design hooks/use-theme.ts and use-motion-gate.ts own both attributes after
   hydration; keep the key and the attribute values in step. Served as a file, not inline: CSP is
   script-src 'self'. */
(function () {
  var root = document.documentElement;
  try {
    var t = localStorage.getItem("sema-theme");
    if (t === "dark" || t === "light") root.setAttribute("data-theme", t);
  } catch (e) {}
  try {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    root.setAttribute("data-motion", "pending");
    /* If the app never hydrates (module script failed), release a visible page after 8s. */
    setTimeout(function () {
      if (root.getAttribute("data-motion") === "pending" && document.visibilityState === "visible") root.setAttribute("data-motion", "");
    }, 8000);
  } catch (e) {}
})();
