// Records the video of shots.mjs: serves the app, opens it with ?video in Chrome (which needs WebGPU, so a window and
// not headless) and steps it a frame at a time (see src/mapWorld/debug/MapWorldVideo.ts). The frames are the canvas itself,
// without the UI, at the size and frame rate of shots.mjs however fast the machine is.
//
//   npm run video                           all shots, then joined into video/out/raytracedMap.mp4
//   npm run video -- --shots tokyo,london   only these shots again, then joined with the ones recorded before
//   npm run video -- --stills               no video: 6 stills per shot in video/out/stills (--stills=12 for 12)
//   npm run video -- --join                 only joins the recorded shots again (another order or crf)
//
// A shot is recorded lossless to video/out/shots/<name>.mp4, the join encodes them once. Needs ffmpeg on the path.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createServer } from 'vite';
import { shots as allShots, video } from './shots.mjs';

const TIMES_OF_DAY = ['sunrise', 'morning', 'noon', 'afternoon', 'sunset'];
// RGB to the limited range BT.709 of HD video, and says so in the file: without it the colors shift
const COLOR_FILTER = 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p';
const COLOR_TAGS = ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv'];

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'video', 'out');
const shotsDir = path.join(outDir, 'shots');
const stillsDir = path.join(outDir, 'stills');

const args = process.argv.slice(2);
const option = (name) => {
  const arg = args.find(a => a === `--${name}` || a.startsWith(`--${name}=`));
  if (arg === undefined) return undefined;
  const next = args[args.indexOf(arg) + 1];
  return arg.includes('=') ? arg.slice(arg.indexOf('=') + 1) : next && !next.startsWith('--') ? next : '';
};
const stills = option('stills') !== undefined ? parseInt(option('stills')) || 6 : 0;
const joinOnly = option('join') !== undefined;
const names = option('shots') ? option('shots').split(',') : null;
for (const name of names ?? []) {
  if (!allShots.some(s => s.name === name)) throw new Error(`no shot ${name} in shots.mjs`);
}
const shots = names ? allShots.filter(s => names.includes(s.name)) : allShots;
const shotFile = (shot) => path.join(shotsDir, `${shot.name}.mp4`);

function runFfmpeg(ffmpegArgs, stdin = 'ignore') {
  const process = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', ...ffmpegArgs], { stdio: [stdin, 'inherit', 'inherit'] });
  const done = once(process, 'close').then(([code]) => {
    if (code !== 0) throw new Error(`ffmpeg exited with ${code}`);
  });
  return { process, done };
}

// what the page posts to /__frame goes here
let onFrame = async (_image, _index) => {};

async function startServer() {
  const server = await createServer({
    // not vite.config.js: its https (mkcert) is for phones on the network, localhost has WebGPU over http
    configFile: false,
    root,
    base: './',
    logLevel: 'warn',
    cacheDir: 'node_modules/.vite-video',
    resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
    server: { host: 'localhost', port: 5199 },
    plugins: [{
      name: 'video-frames',
      configureServer(vite) {
        vite.middlewares.use('/__frame', (req, res) => {
          const chunks = [];
          req.on('data', chunk => chunks.push(chunk));
          req.on('end', async () => {
            try {
              await onFrame(Buffer.concat(chunks), parseInt(req.headers['x-frame']));
              res.end('ok');
            } catch (e) {
              console.error(e);
              res.statusCode = 500;
              res.end(String(e));
            }
          });
        });
      },
    }],
  });
  await server.listen();
  return server;
}

async function openApp(url) {
  const browser = await puppeteer.launch({
    channel: 'chrome',
    headless: false,
    defaultViewport: null,
    // a shot is 1 call
    protocolTimeout: 3600_000,
    args: [
      '--window-size=1280,800',
      // keeps going at full speed behind other windows
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    ],
  });
  const page = (await browser.pages())[0] ?? await browser.newPage();
  page.on('pageerror', e => console.error('[page]', e.message));
  page.on('requestfailed', r => console.error('[page]', r.url(), r.failure()?.errorText));
  page.on('console', m => {
    // not the favicon the page doesn't have
    if (m.type() === 'error' && !m.text().includes('404')) console.error('[page]', m.text());
  });
  // the first city doesn't matter, every shot loads its own
  await page.goto(`${url}?video&city=Amsterdam`);
  await page.waitForFunction(() => window.__mapWorldVideo?.ready, { timeout: 120_000 });
  await page.evaluate(({ width, height, fps, params }) => window.__mapWorldVideo.setup(width, height, fps, params), video);
  return { browser, page };
}

async function record(page, shot) {
  const timeOfDay = TIMES_OF_DAY.indexOf(shot.timeOfDay);
  if (timeOfDay < 0) throw new Error(`${shot.name}: no timeOfDay ${shot.timeOfDay}`);
  const frames = Math.round(shot.seconds * video.fps);
  const start = Date.now();
  process.stdout.write(`${shot.name}: loading`);
  const waited = await page.evaluate(
    (s, leadIn) => window.__mapWorldVideo.loadShot(s, leadIn),
    { ...shot, params: { ...shot.params, timeOfDay } }, video.leadIn,
  );
  if (waited < 0) process.stdout.write(' (the map did not settle, recording anyway)');

  let received = 0;
  let ffmpeg = null;
  if (stills) {
    onFrame = async (image, index) => {
      await fs.promises.writeFile(path.join(stillsDir, `${shot.name}-${String(index).padStart(4, '0')}.jpg`), image);
    };
  } else {
    ffmpeg = runFfmpeg([
      '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${video.width}x${video.height}`, '-framerate', String(video.fps), '-i', '-',
      '-vf', COLOR_FILTER, '-c:v', 'libx264', '-preset', 'veryfast', '-qp', '0', ...COLOR_TAGS, shotFile(shot),
    ], 'pipe');
    onFrame = async (image) => {
      if (!ffmpeg.process.stdin.write(image)) await once(ffmpeg.process.stdin, 'drain');
      if (++received % 30 === 0) process.stdout.write(`\r${shot.name}: frame ${received} / ${frames}   `);
    };
  }

  // every frame of the shot is stepped, also for stills: the lightmap and the TAA are as in the video
  const inside = await page.evaluate(async (frames, stills) => {
    const v = window.__mapWorldVideo;
    const post = async (blob, index) => {
      const response = await fetch('/__frame', { method: 'POST', body: blob, headers: { 'x-frame': String(index) } });
      if (!response.ok) throw new Error(await response.text());
    };
    if (stills) {
      let done = 0;
      for (let i = 0; i < stills; i++) {
        const index = Math.round(i * (frames - 1) / Math.max(stills - 1, 1));
        await v.skip(index - done);
        await post(await v.frame('image/jpeg', .92), index);
        done = index + 1;
      }
    } else {
      // the pixels as they are: ffmpeg encodes them
      for (let i = 0; i < frames; i++) await post(await v.framePixels(), i);
    }
    return v.insideFrames;
  }, frames, stills);

  if (ffmpeg) {
    ffmpeg.process.stdin.end();
    await ffmpeg.done;
  }
  process.stdout.write(`\r${shot.name}: ${stills ? `${stills} stills` : `${frames} frames`} in ${((Date.now() - start) / 1000).toFixed(0)} s      \n`);
  if (inside > 0) console.warn(`  ! the camera is inside a building in ${inside} frames`);
}

async function join() {
  const files = allShots.filter(shot => {
    const exists = fs.existsSync(shotFile(shot));
    if (!exists) console.warn(`  ! ${shot.name} is not recorded yet, left out`);
    return exists;
  });
  if (files.length === 0) return;
  const list = path.join(shotsDir, 'list.txt');
  fs.writeFileSync(list, files.map(shot => `file '${shot.name}.mp4'\n`).join(''));
  const out = path.join(outDir, 'raytracedMap.mp4');
  await runFfmpeg([
    '-f', 'concat', '-safe', '0', '-i', list,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', String(video.crf), '-profile:v', 'high', '-pix_fmt', 'yuv420p',
    ...COLOR_TAGS, '-movflags', '+faststart', '-an', out,
  ]).done;
  const seconds = files.reduce((sum, shot) => sum + shot.seconds, 0);
  console.log(`${out}: ${files.length} shots, ${seconds} s, ${(fs.statSync(out).size / 1048576).toFixed(1)} MB`);
}

fs.mkdirSync(shotsDir, { recursive: true });
if (stills) fs.mkdirSync(stillsDir, { recursive: true });

if (!joinOnly) {
  const server = await startServer();
  const { browser, page } = await openApp(server.resolvedUrls.local[0]);
  try {
    for (const shot of shots) await record(page, shot);
  } finally {
    await browser.close();
    await server.close();
  }
}
if (!stills) await join();
