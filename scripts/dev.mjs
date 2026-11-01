#!/usr/bin/env node
// No frontend build: serve the existing UI and WebSocket API from one origin.
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import http from "node:http";

const root = fileURLToPath(new URL("../", import.meta.url));
const args = process.argv.slice(2);
let host = "127.0.0.1";
let port = 3000;
let open = process.platform !== "linux" || Boolean(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);
let setupOnly = false;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--host") host = args[++i];
  else if (args[i] === "--port") port = Number(args[++i]);
  else if (args[i] === "--open") open = true;
  else if (args[i] === "--no-open") open = false;
  else if (args[i] === "--setup") setupOnly = true;
  else if (args[i] === "--help") {
    console.log("npm run dev -- [--host 127.0.0.1] [--port 3000] [--open | --no-open]");
    console.log("Python 3.10+ is required. Missing dependencies are installed in .venv automatically.");
    console.log("Set VLVC_PYTHON to use a specific Python environment. npm run setup refreshes dependencies.");
    process.exit(0);
  } else {
    console.error(`Unknown argument: ${args[i]}. Use npm run dev -- --help.`);
    process.exit(1);
  }
}
if (!host || !Number.isInteger(port) || port < 1 || port > 65535) {
  console.error("A host and a port between 1 and 65535 are required.");
  process.exit(1);
}

function run(python, command, options = {}) {
  return spawnSync(python.command, [...python.prefix, ...command], { cwd: root, ...options });
}

function required(python, command) {
  const result = run(python, command, { stdio: "inherit" });
  if (result.error || result.status !== 0) {
    throw new Error(result.error?.message || `Python command failed (exit ${result.status})`);
  }
}

function findPython() {
  const local = join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const active = process.env.VIRTUAL_ENV
    ? join(process.env.VIRTUAL_ENV, process.platform === "win32" ? "Scripts/python.exe" : "bin/python") : null;
  const candidates = process.env.VLVC_PYTHON
    ? [{ command: process.env.VLVC_PYTHON, prefix: [] }]
    : [
      ...[active, local].filter((p) => p && existsSync(p)).map((command) => ({ command, prefix: [] })),
      { command: "python3", prefix: [] }, { command: "python", prefix: [] },
      ...(process.platform === "win32" ? [{ command: "py", prefix: ["-3"] }] : []),
    ];
  for (const python of candidates) {
    const result = run(python, ["-c", "import sys; assert sys.version_info >= (3, 10)"]);
    if (result.status === 0) return python;
  }
  throw new Error("Python 3.10+ was not found. Install Python or set VLVC_PYTHON to its executable.");
}

function prepare(python) {
  // With an explicit/activated environment, use it. Otherwise install into a
  // repository-local venv rather than modifying the global Python installation.
  if (!process.env.VLVC_PYTHON && !process.env.VIRTUAL_ENV) {
    const local = join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
    if (!existsSync(local)) required(python, ["-m", "venv", join(root, ".venv")]);
    python = { command: local, prefix: [] };
  }
  console.log("Preparing Python dependencies (first setup may take a few minutes)…");
  if (run(python, ["-c", "import torch"]).status !== 0) {
    // A CPU default avoids downloading CUDA libraries on machines without a GPU.
    // An already-installed CUDA build is preserved.
    required(python, ["-m", "pip", "install", "torch>=2.6", "--index-url", "https://download.pytorch.org/whl/cpu"]);
  }
  required(python, ["-m", "pip", "install", "-e", ".[engine,web]"]);
  return python;
}

try {
  let python = findPython();
  const probe = ["-c", "import torch, numpy, scipy, soundfile, onnxruntime, fastapi, uvicorn, multipart, websockets"];
  if (setupOnly || run(python, probe).status !== 0) python = prepare(python);
  if (setupOnly) {
    console.log("Setup complete. Run npm run dev to open the Web UI.");
    process.exit(0);
  }
  const browserHost = host === "::" ? "::1" : host === "0.0.0.0" ? "127.0.0.1" : host;
  const url = `http://${browserHost.includes(":") ? `[${browserHost}]` : browserHost}:${port}`;
  console.log(`\nVL VoiceChanger dev: ${url}`);
  console.log("Python changes reload automatically. Refresh the browser after UI changes. Ctrl+C to stop.\n");
  const child = spawn(python.command, [...python.prefix,
    "-m", "uvicorn", "engine.web.server:app", "--reload", "--reload-dir", resolve(root, "engine"),
    "--host", host, "--port", String(port), "--ws-max-size", "6400", "--ws-max-queue", "4",
  ], { cwd: root, stdio: "inherit", detached: process.platform !== "win32",
    env: { ...process.env, PYTHONUNBUFFERED: "1" } });
  let timer;
  let stopTimer;
  let opening = false;
  let stopping = false;
  function stop() {
    if (stopping) return;
    stopping = true;
    if (process.platform === "win32") {
      spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
    } else {
      child.kill("SIGINT");
      stopTimer = setTimeout(() => {
        try { process.kill(-child.pid, "SIGTERM"); } catch (_) { /* already stopped */ }
      }, 5000);
      stopTimer.unref();
    }
  }
  function cleanup() {
    if (process.platform !== "win32" && child.exitCode === null) {
      try { process.kill(-child.pid, "SIGTERM"); } catch (_) { /* already stopped */ }
    }
  }
  process.on("exit", cleanup);
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  // Wait for our server to be ready before launching the browser.
  if (open) timer = setInterval(() => {
    if (opening) return;
    opening = true;
    const request = http.get(`${url}/api/health`, { timeout: 1000 }, (response) => {
      response.resume();
      if (response.statusCode !== 200) { opening = false; return; }
      clearInterval(timer);
      const browser = spawn(python.command, [...python.prefix, "-m", "webbrowser", "-t", url],
        { cwd: root, stdio: "ignore" });
      browser.on("error", () => console.log(`Open ${url} in your browser.`));
    });
    request.on("timeout", () => request.destroy());
    request.on("error", () => { opening = false; });
  }, 400);
  child.on("error", (error) => { clearInterval(timer); console.error(error.message); process.exitCode = 1; });
  child.on("exit", (code, signal) => {
    clearInterval(timer);
    clearTimeout(stopTimer);
    process.removeListener("exit", cleanup);
    process.removeListener("SIGINT", stop);
    process.removeListener("SIGTERM", stop);
    process.exitCode = code ?? (signal === "SIGINT" || signal === "SIGTERM" ? 0 : 1);
  });
} catch (error) {
  console.error(`\nDev startup failed: ${error.message}`);
  process.exitCode = 1;
}
