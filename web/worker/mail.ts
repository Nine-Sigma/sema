/* Outbound mail over SMTP (Titan) from a Worker: worker-mailer speaks SMTP on a
   `cloudflare:sockets` TCP connection. Port 465 = implicit TLS, 587 = STARTTLS. */

import { WorkerMailer } from "worker-mailer";
import type { Env } from "./env";

export type Mail = { subject: string; text: string; replyTo?: string };

export async function sendMail(env: Env, mail: Mail): Promise<void> {
  if (!env.SMTP_PASSWORD) throw new Error("SMTP_PASSWORD secret not set");
  const port = Number(env.SMTP_PORT);
  await WorkerMailer.send(
    {
      host: env.SMTP_HOST,
      port,
      secure: port === 465,
      startTls: port !== 465,
      credentials: { username: env.SMTP_USER, password: env.SMTP_PASSWORD },
      authType: ["plain", "login"],
    },
    { from: env.FROM_ADDRESS, to: env.TO_ADDRESS, subject: mail.subject, text: mail.text, reply: mail.replyTo },
  );
}
