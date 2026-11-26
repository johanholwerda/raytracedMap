// temporary: loads the URL camera and reports the eye height over the ground, screenshots it
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = process.argv[2];
const server = await createServer({
  configFile: false, root, base: './', logLevel: 'warn', cacheDir: 'node_modules/.vite-video',
  resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
  server: { host: 'localhost', port: 5213 },
});
await server.listen();
const browser = await puppeteer.launch({ channel: 'chrome', headless: false, defaultViewport: null,
  args: ['--window-size=1280,800', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
const page = (await browser.pages())[0];
page.on('pageerror', e => console.error('[page]', e.message));
await page.goto('http://localhost:5213/?tileTimings&lat=46.005779&lng=7.732974&dist=921&pitch=-11.46&yaw=-63.60&time=afternoon');
await new Promise(r => setTimeout(r, 25000));
const report = () => page.evaluate(() => {
  const w = window.__mapWorld;
  const c = w.camera;
  const eye = c.worldPosition;
  const g = w.heightField.heightAt(eye.x, eye.z);
  const m = w.projection.metersToNorm;
  return { userPitch: c.rotationController.eulerXY.x, eyeAboveGroundM: (eye.y - g) / m, pivotY: c.pivot.position.y / m };
});
console.log(await report());
await page.screenshot({ path: out });
await page.evaluate(() => { window.__mapWorld.camera.constrainView = null; window.__mapWorld.camera.pivot.position = window.__mapWorld.camera.pivot.position.clone(); });
await new Promise(r => setTimeout(r, 3000));
console.log('off', await report());
await page.screenshot({ path: out.replace('.png', '-off.png') });
await browser.close();
await server.close();
