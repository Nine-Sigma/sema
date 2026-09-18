import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { Landing } from "./landing";

/* QA convenience: ?theme=dark|light pins the theme before first paint. */
const pinned = new URLSearchParams(location.search).get("theme");
if (pinned === "dark" || pinned === "light") document.documentElement.setAttribute("data-theme", pinned);

const root = document.getElementById("root");
if (!root) throw new Error("#root missing");
createRoot(root).render(
  <React.StrictMode>
    <Landing />
  </React.StrictMode>,
);
