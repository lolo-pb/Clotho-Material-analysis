const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const net = require("node:net");
const path = require("node:path");

let backendProcess;

function reservePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(error => (error ? reject(error) : resolve(port)));
    });
  });
}

function backendCommand(port, token) {
  const args = ["--port", String(port), "--auth-token", token];
  if (app.isPackaged) {
    return {
      command: path.join(process.resourcesPath, "backend", "clotho-backend.exe"),
      args,
    };
  }

  return {
    command: process.env.CLOTHO_PYTHON || "python",
    args: [path.join(__dirname, "..", "web.py"), ...args],
  };
}

function startBackend(port, token) {
  const { command, args } = backendCommand(port, token);
  backendProcess = spawn(command, args, { windowsHide: true });
  backendProcess.stderr.on("data", data => console.error(`backend: ${data}`));
  backendProcess.on("error", error => console.error("Could not start backend:", error));
}

async function waitForBackend(url, token) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (backendProcess.exitCode !== null) {
      throw new Error("The analysis backend exited during startup.");
    }
    try {
      const response = await fetch(`${url}/health`, {
        headers: { "X-Clotho-Session": token },
      });
      if (response.ok) return;
    } catch {
      // The server has not started listening yet.
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error("The analysis backend did not start within 30 seconds.");
}

function stopBackend() {
  if (backendProcess && backendProcess.exitCode === null) backendProcess.kill();
}

async function openWindow() {
  const port = await reservePort();
  const token = crypto.randomBytes(32).toString("hex");
  const url = `http://127.0.0.1:${port}`;
  startBackend(port, token);
  await waitForBackend(url, token);

  const window = new BrowserWindow({
    width: 1200,
    height: 850,
    minWidth: 760,
    minHeight: 600,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  await window.loadURL(`${url}/?token=${encodeURIComponent(token)}`);
}

app.whenReady().then(openWindow).catch(error => {
  stopBackend();
  dialog.showErrorBox("Clotho could not start", error.message);
  app.quit();
});

app.on("window-all-closed", () => app.quit());
app.on("before-quit", stopBackend);
