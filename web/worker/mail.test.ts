import { beforeEach, describe, expect, it, vi } from "vitest";
import { WorkerMailer } from "worker-mailer";
import { sendMail } from "./mail";
import type { Env } from "./env";

vi.mock("worker-mailer", () => ({ WorkerMailer: { send: vi.fn(async () => undefined) } }));
const send = vi.mocked(WorkerMailer.send);

const env = (port: string, password?: string) =>
  ({ SMTP_HOST: "smtp.titan.email", SMTP_PORT: port, SMTP_USER: "dean@withsema.ai", SMTP_PASSWORD: password, FROM_ADDRESS: "waitlist@withsema.ai", TO_ADDRESS: "waitlist@withsema.ai" }) as unknown as Env;

describe("sendMail", () => {
  beforeEach(() => send.mockClear());

  it("throws when the SMTP_PASSWORD secret is missing", async () => {
    await expect(sendMail(env("465"), { subject: "s", text: "t" })).rejects.toThrow("SMTP_PASSWORD");
    expect(send).not.toHaveBeenCalled();
  });

  it("uses implicit TLS on 465 and passes credentials, addresses and reply-to", async () => {
    await sendMail(env("465", "pw"), { subject: "Waitlist: a@b.co", text: "body", replyTo: "a@b.co" });
    const [opts, mail] = send.mock.calls[0]!;
    expect(opts).toMatchObject({ host: "smtp.titan.email", port: 465, secure: true, startTls: false, credentials: { username: "dean@withsema.ai", password: "pw" } });
    expect(mail).toMatchObject({ from: "waitlist@withsema.ai", to: "waitlist@withsema.ai", subject: "Waitlist: a@b.co", text: "body", reply: "a@b.co" });
  });

  it("uses STARTTLS on 587", async () => {
    await sendMail(env("587", "pw"), { subject: "s", text: "t" });
    expect(send.mock.calls[0]![0]).toMatchObject({ port: 587, secure: false, startTls: true });
  });
});
