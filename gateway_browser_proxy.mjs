import { chromium } from "playwright-core";
import fs from "fs";

function parseArgs() {
  const raw = process.argv[2];
  if (!raw) throw new Error("missing payload");
  return JSON.parse(raw);
}

function resolveExecutable() {
    const candidates = [
        process.env.PW_CHROME_PATH,
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ].filter(Boolean);
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) return candidate;
    }
    return null;
}

function normalizeChoiceJson(rawText) {
  try {
    const parsed = JSON.parse(rawText);
    const content = parsed?.choices?.[0]?.message?.content;
    if (typeof content !== "string") return rawText;
    const trimmed = content.trim();
    if (!trimmed || !/^[\[{]/.test(trimmed)) return rawText;
    JSON.parse(trimmed);
    return JSON.stringify({
      ...parsed,
      choices: [
        {
          ...(parsed.choices?.[0] || {}),
          message: {
            ...(parsed.choices?.[0]?.message || {}),
            content: trimmed,
          },
        },
      ],
    });
  } catch {
    return rawText;
  }
}

async function main() {
  const payload = parseArgs();
  if (payload.body?.response_format?.type === "json_object") {
    delete payload.body.response_format;
  }
  const executablePath = resolveExecutable();
  const launchOptions = { headless: true };
  if (executablePath) {
    launchOptions.executablePath = executablePath;
  }
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const originBase = String(payload.base_url || "").replace(/\/v1\/?$/i, "");
    let navError = null;
    for (let i = 0; i < 2; i++) {
      try {
        await page.goto(originBase + "/login", { waitUntil: "domcontentloaded", timeout: 120000 });
        navError = null;
        break;
      } catch (err) {
        navError = err;
        await page.waitForTimeout(1200);
      }
    }
    if (navError) throw navError;
    await page.waitForTimeout(1200);
    const result = await page.evaluate(async (input) => {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer " + input.api_key,
        },
        body: JSON.stringify(input.body),
      });
      const text = await resp.text();
      return { ok: resp.ok, status: resp.status, text };
    }, payload);
    if (payload.force_json && result && typeof result.text === "string") {
      result.text = normalizeChoiceJson(result.text);
    }
    process.stdout.write(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err));
  process.exit(1);
});
