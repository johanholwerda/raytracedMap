/// <reference types="vite/types/importMeta.d.ts" />
/**
 * ?remote on the dev server: the page reports to tools/remoteDebugPlugin.js over Vite's HMR websocket (console, errors,
 * WebGPU errors, device loss, shader compile messages, fps every second) and runs the JavaScript tools/remote.mjs sends,
 * for debugging and timing on a device without devtools (an iPhone on Windows). Installed before anything else runs,
 * so it wraps requestDevice and createShaderModule.
 */
const MAX_TEXT = 20000;

function describe(value: unknown, depth = 0, seen = new WeakSet<object>()): unknown {
  if (value === null || value === undefined) return value;
  const type = typeof value;
  if (type === 'number' || type === 'string' || type === 'boolean') return value;
  if (type === 'bigint') return value.toString();
  if (type === 'function') return `[function ${(value as Function).name}]`;
  if (type === 'symbol') return value.toString();
  const obj = value as object;
  if (obj instanceof Error) return `${obj.name}: ${obj.message}\n${obj.stack ?? ''}`;
  if (seen.has(obj)) return '[circular]';
  if (depth > 4) return `[${obj.constructor?.name ?? 'object'}]`;
  seen.add(obj);
  if (obj instanceof Map) return Object.fromEntries(Array.from(obj).map(([k, v]) => [String(k), describe(v, depth + 1, seen)]));
  if (ArrayBuffer.isView(obj)) return `[${obj.constructor.name} length ${(obj as any).length}]`;
  if (Array.isArray(obj)) return obj.slice(0, 200).map((v) => describe(v, depth + 1, seen));
  // Set and the set-likes of WebGPU (GPUSupportedFeatures)
  if (Symbol.iterator in obj) return Array.from(obj as Iterable<unknown>).slice(0, 200).map((v) => describe(v, depth + 1, seen));
  const out: Record<string, unknown> = {};
  let count = 0;
  // for..in: WebIDL attributes (GPUSupportedLimits, GPUAdapterInfo) are getters on the prototype
  for (const key in obj) {
    if (++count > 200) break;
    try {
      const v = (obj as any)[key];
      if (typeof v !== 'function') out[key] = describe(v, depth + 1, seen);
    } catch (e) {
      out[key] = '[throws]';
    }
  }
  return out;
}

function toText(args: unknown[]): string {
  const text = args.map((a) => typeof a === 'string' ? a : JSON.stringify(describe(a))).join(' ');
  return text.length > MAX_TEXT ? text.slice(0, MAX_TEXT) + '…' : text;
}

export function installRemoteDebug() {
  const hot = import.meta.hot;
  if (!hot || !new URLSearchParams(window.location.search).has('remote')) return;

  const id = Math.random().toString(36).slice(2, 8);
  let queue: { level: string, text: string }[] = [];
  const send = (level: string, text: string) => {
    queue.push({ level, text });
  };
  setInterval(() => {
    if (queue.length === 0) return;
    hot.send('remote:log', { id, entries: queue });
    queue = [];
  }, 250);

  for (const level of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    const original = console[level].bind(console);
    console[level] = (...args: unknown[]) => {
      original(...args);
      try {
        send(level, toText(args));
      } catch (e) {
        // never let reporting break the page
      }
    };
  }
  window.addEventListener('error', (e) => send('uncaught', `${e.message} at ${e.filename}:${e.lineno}:${e.colno}\n${e.error?.stack ?? ''}`));
  window.addEventListener('unhandledrejection', (e) => send('unhandledrejection', toText([e.reason])));

  // WebGPU: every validation/out-of-memory/internal error, device loss and shader compile message
  if (typeof GPUAdapter !== 'undefined') {
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
      send('gpu', `adapter info ${toText([(this as any).info])}`);
      send('gpu', `adapter features ${toText([this.features])}`);
      send('gpu', `adapter limits ${toText([this.limits])}`);
      send('gpu', `requestDevice ${toText([descriptor])}`);
      const device = await requestDevice.call(this, descriptor);
      (window as any).__gpuDevice = device;
      let errorCount = 0;
      device.addEventListener('uncapturederror', (event) => {
        const error = (event as GPUUncapturedErrorEvent).error;
        // a broken pass errors every frame: report the first 50, then every 500th
        if (++errorCount <= 50 || errorCount % 500 === 0) {
          send('gpuerror', `#${errorCount} ${error.constructor.name}: ${error.message}`);
        }
      });
      device.lost.then((info) => send('gpulost', `${info.reason}: ${info.message}`));
      return device;
    };
    const createShaderModule = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (this: GPUDevice, descriptor: GPUShaderModuleDescriptor) {
      const module = createShaderModule.call(this, descriptor);
      module.getCompilationInfo?.().then((info) => {
        const messages = info.messages.filter((m) => m.type !== 'info');
        if (messages.length === 0) return;
        const lines = descriptor.code.split('\n');
        send('shader', `${descriptor.label ?? 'shader'}:\n` + messages.map((m) =>
          `${m.type} ${m.lineNum}:${m.linePos} ${m.message}\n    ${lines[m.lineNum - 1]?.trim() ?? ''}`).join('\n'));
      });
      return module;
    };
  }

  // fps and the worst frame of every second
  let frames = 0;
  let worst = 0;
  let last = performance.now();
  let secondStart = last;
  const tick = () => {
    const now = performance.now();
    frames++;
    worst = Math.max(worst, now - last);
    last = now;
    if (now - secondStart >= 1000) {
      hot.send('remote:stats', {
        id, url: location.href,
        stats: { fps: Math.round(frames * 1000 / (now - secondStart)), worstFrameMs: Math.round(worst), w: innerWidth, h: innerHeight, dpr: devicePixelRatio },
      });
      frames = 0;
      worst = 0;
      secondStart = now;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  // code from tools/remote.mjs: the body of an async function, its return value comes back
  const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
  hot.on('remote:eval', async (data: { target: string, evalId: number, code: string }) => {
    if (data.target !== id) return;
    let reply: Record<string, unknown>;
    try {
      const fn = new AsyncFunction(data.code);
      reply = { result: describe(await fn()) };
    } catch (e) {
      reply = { error: toText([e]) };
    }
    hot.send('remote:result', { id, evalId: data.evalId, ...reply });
  });

  const hello = () => hot.send('remote:hello', { id, ua: navigator.userAgent, url: location.href });
  hello();
  // the server restarted (vite.config.js changed): introduce ourselves again
  hot.on('vite:ws:connect', hello);
}
