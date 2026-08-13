const { chromium } = require("C:/Users/46884/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

(async () => {
  const userDataDir = "C:/Users/46884/AppData/Local/Google/Chrome/User Data";
  const context = await chromium.launchPersistentContext(userDataDir, {
    executablePath: "C:/Users/46884/AppData/Local/Google/Chrome/Application/chrome.exe",
    headless: true,
    args: [
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--disable-background-networking",
    ],
    timeout: 30000,
  });
  console.log("LAUNCHED");
  const page = context.pages()[0] || (await context.newPage());
  await page.goto("https://data.cma.cn/", { waitUntil: "domcontentloaded", timeout: 45000 }).catch((e) => console.log("goto:", e.message));
  await page.waitForTimeout(8000);
  console.log("URL:", page.url());
  console.log("TITLE:", await page.title());
  const ls = await page.evaluate(() => {
    const o = {};
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      o[k] = localStorage.getItem(k);
    }
    return o;
  });
  console.log("LOCALSTORAGE:", JSON.stringify(ls).slice(0, 3000));
  const login = await page.evaluate(() => {
    const el = document.querySelector("#loginStatus");
    return el ? el.innerText.trim() : "";
  });
  console.log("LOGIN:", login || "no #loginStatus");
  await context.close();
})().catch((e) => {
  console.error("ERR", e);
  process.exit(1);
});
