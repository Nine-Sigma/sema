/* Applies the saved theme before the stylesheet loads so a pinned theme never flashes.
   Keep the key in step with @sema/design hooks/use-theme.ts. Served as a file, not inline: CSP is script-src 'self'. */
(function () {
  try {
    var t = localStorage.getItem("sema-theme");
    if (t === "dark" || t === "light") document.documentElement.setAttribute("data-theme", t);
  } catch (e) {}
})();
