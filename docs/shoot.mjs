// Capture README screenshots from the running app (node docs/shoot.mjs).
// Drives headless Chrome over CDP; no npm dependencies (node >= 22 has WebSocket).
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const APP = process.env.APP_URL || "http://127.0.0.1:8000/";
const ROOT = decodeURIComponent(new URL("../", import.meta.url).pathname).replace(/^\//, "");
const OUT = `${ROOT}docs/images/`;
const CHROME = process.env.CHROME ||
  "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9223;
const PROFILE = `${ROOT}docs/.chrome-profile`;

mkdirSync(OUT, { recursive: true });
rmSync(PROFILE, { recursive: true, force: true });

const chrome = spawn(CHROME, [
  "--headless=new", "--no-sandbox", `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  "--hide-scrollbars", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--force-device-scale-factor=1", "about:blank",
], { stdio: ["ignore", "pipe", "pipe"] });
chrome.on("error", (e) => console.error("chrome spawn failed:", e.code));
let chromeErr = "";
chrome.stderr.on("data", (d) => (chromeErr += d));
chrome.on("exit", (code) => {
  if (code) console.error(`chrome exited ${code}: ${chromeErr.slice(0, 400)}`);
});

async function connect() {
  for (let i = 0; i < 80; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      if (res.ok) {
        const pages = (await res.json()).filter((t) => t.type === "page");
        if (pages.length) return pages[0].webSocketDebuggerUrl;
      }
    } catch {}
    await sleep(250);
  }
  throw new Error(`chrome did not expose a debug port (${chromeErr.slice(0, 200)})`);
}

class Page {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      } else if (msg.method) {
        this.events.push(msg);
      }
    };
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params, sessionId }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  async waitFor(method, timeout = 15000) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      const hit = this.events.findIndex((e) => e.method === method);
      if (hit >= 0) return this.events.splice(hit, 1)[0].params;
      await sleep(100);
    }
    throw new Error(`timeout waiting for ${method}`);
  }
  async goto(url, { width, height }) {
    await this.send("Emulation.setDeviceMetricsOverride", {
      width, height, deviceScaleFactor: 2, mobile: width < 600,
    });
    await this.send("Page.navigate", { url });
    await this.waitFor("Page.loadEventFired", 20000);
    await sleep(1200);
  }
  async clickText(text) {
    const expr = `(() => {
      const el = [...document.querySelectorAll('button, a')]
        .find(n => (n.innerText || '').toLowerCase().includes(${JSON.stringify(text.toLowerCase())}));
      if (!el) return 'missing';
      el.click();
      return 'clicked';
    })()`;
    const { result } = await this.send("Runtime.evaluate", { expression: expr, returnByValue: true });
    if (result.value !== "clicked") {
      const dump = await this.send("Runtime.evaluate", {
        expression: "document.body.innerText.slice(0, 400)", returnByValue: true,
      });
      throw new Error(`no button matching ${text}; page says: ${dump.result.value}`);
    }
    await sleep(900);
  }
  async focus(selector) {
    const expr = `(() => {
      const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) return 'missing';
      el.focus();
      return 'focused';
    })()`;
    const { result } = await this.send("Runtime.evaluate", { expression: expr, returnByValue: true });
    if (result.value !== "focused") throw new Error(`no input matching ${selector}`);
  }
  async insertText(text) {
    await this.send("Input.insertText", { text });
  }
  async pressEnter() {
    for (const type of ["rawKeyDown", "char", "keyUp"]) {
      await this.send("Input.dispatchKeyEvent", {
        type, key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13,
        text: type === "char" ? "\r" : undefined,
      });
    }
  }
  async waitForText(text, timeout = 40000) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      const { result } = await this.send("Runtime.evaluate", {
        expression: `document.body.innerText.includes(${JSON.stringify(text)})`,
        returnByValue: true,
      });
      if (result.value) return;
      await sleep(400);
    }
    throw new Error(`timeout waiting for text: ${text}`);
  }
  async shot(name, { full = false, quality = 0 } = {}) {
    const format = quality ? "jpeg" : "png";
    const params = { format, captureBeyondViewport: full };
    if (quality) params.quality = quality;
    const { data } = await this.send("Page.captureScreenshot", params);
    const file = `docs/images/${name}.${format === "jpeg" ? "jpg" : "png"}`;
    writeFileSync(`${OUT}${name}.${format === "jpeg" ? "jpg" : "png"}`, Buffer.from(data, "base64"));
    console.log("wrote", file);
  }
}

const browserWs = await connect();
const ws = new WebSocket(browserWs);
await new Promise((r) => (ws.onopen = r));
const page = new Page(ws);
await page.send("Page.enable");
await page.send("Runtime.enable");

try {
  await page.goto(APP, { width: 1440, height: 900 });
  await sleep(2200);                       // the cover reveals with a tween
  await page.shot("cover", { quality: 88 });

  await page.clickText("start writing");
  await sleep(1600);
  await page.focus("input");
  await page.insertText("harbor");
  await page.pressEnter();
  await page.waitForText("the poster", 45000);   // the button only exists once a poem arrived
  await page.waitForText("harbor", 20000);       // the answer card has typed itself out
  await sleep(2600);
  await page.shot("write", { quality: 88 });

  await page.clickText("the poster");
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: 1440, height: 1000, deviceScaleFactor: 1.4, mobile: false,
  });
  await sleep(2500);
  // the poster view reveals its copy on scroll, which does not settle reliably in a headless
  // capture, so the desk is shot in sheet view instead
  await page.clickText("sheet");
  await sleep(2200);
  await page.shot("sheet");

  await page.goto(APP, { width: 390, height: 844 });
  await sleep(2200);
  await page.shot("mobile", { quality: 88 });
} finally {
  ws.close();
  chrome.kill();
}
