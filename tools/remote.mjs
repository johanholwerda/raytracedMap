#!/usr/bin/env node
/**
 * Talks to the page on a remote device (opened with ?remote) through the dev server, see tools/remoteDebugPlugin.js.
 *
 *   node tools/remote.mjs clients                 the connected pages, their fps
 *   node tools/remote.mjs log [since]             the log after sequence number `since`
 *   node tools/remote.mjs tail                    follow the log
 *   node tools/remote.mjs eval "return 1+1"       run the body of an async function on the page, print the result
 *   node tools/remote.mjs eval -f script.js       the same from a file
 *   node tools/remote.mjs clear                   empty the log
 *   node tools/remote.mjs screenshot out.jpg      the next frame of the canvas (through the P key screenshot of MapWorld)
 *
 * Options: --ua iPhone (the most recent page whose user agent has it; default the most recent page), --timeout 60,
 * --port 5173.
 */
import fs from 'fs';

// the mkcert certificate of the dev server isn't in Node's CA list
process.emitWarning = () => {};
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const value = args[i + 1];
  args.splice(i, 2);
  return value;
};
const ua = option('--ua', '');
const timeout = option('--timeout', '60');
const port = option('--port', '5173');
const file = option('-f', null);
const base = `https://localhost:${port}/__remote`;
const [command, ...rest] = args;

const get = async (path) => (await fetch(base + path)).json();
const format = (l) => `${l.seq} ${new Date(l.t).toISOString().slice(11, 23)} [${l.client}] ${l.level}: ${l.text}`;

if (command === 'clients') {
  console.log(JSON.stringify(await get('/clients'), null, 2));
} else if (command === 'log') {
  const { logs } = await get(`/log?since=${rest[0] ?? 0}`);
  for (const l of logs) console.log(format(l));
} else if (command === 'tail') {
  let since = 0;
  for (;;) {
    const { seq, logs } = await get(`/log?since=${since}`);
    for (const l of logs) console.log(format(l));
    since = seq;
    await new Promise((r) => setTimeout(r, 500));
  }
} else if (command === 'clear') {
  console.log(JSON.stringify(await get('/clear')));
} else if (command === 'screenshot') {
  // MapWorld saves the canvas right after the frame's submit (a WebGPU canvas is empty in a later task): its toBlob
  // hands over a JPEG instead of a download
  const code = `const mw = window.__mapWorld; const canvas = mw.renderer.canvas;
    return await new Promise((resolve, reject) => {
      const original = HTMLCanvasElement.prototype.toBlob;
      canvas.toBlob = () => {
        delete canvas.toBlob;
        original.call(canvas, (blob) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        }, 'image/jpeg', .85);
      };
      mw.screenshotRequested = true;
      mw.mapRenderer.markDirty();
      setTimeout(() => { delete canvas.toBlob; reject(new Error('no frame in 10 s')); }, 10000);
    });`;
  const res = await fetch(`${base}/eval?ua=${encodeURIComponent(ua)}&timeout=${timeout}`, { method: 'POST', body: code });
  const reply = await res.json();
  if (reply.error) {
    console.error(`[${reply.client ?? '-'}] error: ${reply.error}`);
    process.exitCode = 1;
  } else {
    const out = rest[0] ?? 'screenshot.jpg';
    fs.writeFileSync(out, Buffer.from(reply.result.split(',')[1], 'base64'));
    console.log(`[${reply.client}] ${out}`);
  }
} else if (command === 'eval') {
  const code = file ? fs.readFileSync(file, 'utf8') : rest.join(' ');
  const res = await fetch(`${base}/eval?ua=${encodeURIComponent(ua)}&timeout=${timeout}`, { method: 'POST', body: code });
  const reply = await res.json();
  if (reply.error) {
    console.error(`[${reply.client ?? '-'}] error: ${reply.error}`);
    process.exitCode = 1;
  } else {
    console.log(typeof reply.result === 'string' ? reply.result : JSON.stringify(reply.result, null, 2));
  }
} else {
  console.log('commands: clients | log [since] | tail | clear | screenshot <out.jpg> | eval <code> | eval -f <file>   options: --ua --timeout --port');
}
