/* Bindings and vars declared in wrangler.toml, plus the SMTP_PASSWORD secret
   (`npx wrangler secret put SMTP_PASSWORD`). */

export type Env = {
  ASSETS: Fetcher;
  DB?: D1Database;
  /* Production hostname, e.g. `withsema.ai`. `www.<host>` 301s to it; preview URLs are served as-is. */
  CANONICAL_HOST?: string;
  /* Titan SMTP. SMTP_USER is the mailbox login (dean@) and FROM_ADDRESS must equal it (aliases get 553). */
  SMTP_HOST: string;
  SMTP_PORT: string;
  SMTP_USER: string;
  SMTP_PASSWORD?: string;
  FROM_ADDRESS: string;
  TO_ADDRESS: string;
};
