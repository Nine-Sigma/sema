/* Build-time prerender. `vite build` emits each arm's document with an empty #root; this renders
   the React tree for every arm (src/entry-server.tsx) into that root, so crawlers, link previews
   and readers without JS get the page's text. mount.tsx hydrates it. Also injects the JSON-LD
   block, which needs JSON escaping rather than the HTML escaping the %copy.meta.*% tokens get. */
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { build } from "vite";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = resolve(HERE, "..");
const SSR_DIR = resolve(APP, ".prerender");
const DIST = resolve(APP, "dist");
const SITE_URL = "https://withsema.ai";
const ROOT = '<div id="root"></div>';

type Copy = { meta: { title: string; description: string } };
type Arm = { id: string; path: string; copy: Copy };
type Entry = { render: (copy: Copy) => string; arms: readonly Arm[] };

function jsonLd(copy: Copy): string {
  const org = { "@type": "Organization", "@id": `${SITE_URL}/#org`, name: "Sema", url: `${SITE_URL}/`, email: "waitlist@withsema.ai" };
  const site = { "@type": "WebSite", "@id": `${SITE_URL}/#site`, url: `${SITE_URL}/`, name: "Sema", description: copy.meta.description, publisher: { "@id": `${SITE_URL}/#org` } };
  const data = { "@context": "https://schema.org", "@graph": [org, site] };
  return `<script type="application/ld+json">${JSON.stringify(data).replaceAll("</", "<\\/")}</script>`;
}

async function main(): Promise<void> {
  await build({ configFile: resolve(APP, "vite.config.ts"), logLevel: "warn", build: { ssr: resolve(APP, "src/entry-server.tsx"), outDir: SSR_DIR, emptyOutDir: true } });
  const entry = (await import(pathToFileURL(resolve(SSR_DIR, "entry-server.js")).href)) as Entry;
  for (const arm of entry.arms) {
    const file = resolve(DIST, arm.path.slice(1));
    const html = readFileSync(file, "utf8");
    if (html.split(ROOT).length !== 2) throw new Error(`${file}: expected exactly one ${ROOT}`);
    if (!html.includes("</head>")) throw new Error(`${file}: no </head>`);
    const body = entry.render(arm.copy);
    if (!/<h1[\s>]/.test(body)) throw new Error(`${arm.id}: rendered tree has no h1`);
    const out = html.replace(ROOT, `<div id="root">${body}</div>`).replace("</head>", `    ${jsonLd(entry.arms[0]!.copy)}\n  </head>`);
    writeFileSync(file, out);
    console.log(`prerender: ${arm.path} ${(body.length / 1024).toFixed(0)} KB`);
  }
  rmSync(SSR_DIR, { recursive: true, force: true });
}

await main();
