/**
 * Remote debugging of a page on another device (an iPhone through a tunnel) from this machine: the page (?remote, see
 * src/mapWorld/debug/RemoteDebug.ts) sends its console, errors, WebGPU errors and fps over Vite's HMR websocket, and
 * runs JavaScript sent from tools/remote.mjs, which talks to the endpoints below.
 *
 * The endpoints only answer requests made to localhost directly: through the tunnel (a public URL) they don't exist,
 * else anyone with the URL could run code on the device.
 */
const MAX_LOGS = 5000;

export default function remoteDebugPlugin() {
  return {
    name: 'remote-debug',
    apply: 'serve',
    configureServer(server) {
      /** @type {{seq: number, t: number, client: string, level: string, text: string}[]} */
      const logs = [];
      let seq = 0;
      /** id -> {id, ua, url, connected, lastSeen, stats} */
      const clients = new Map();
      /** eval id -> resolve */
      const pending = new Map();
      let evalId = 0;

      const addLog = (client, level, text) => {
        logs.push({ seq: ++seq, t: Date.now(), client, level, text });
        if (logs.length > MAX_LOGS) logs.splice(0, logs.length - MAX_LOGS);
      };
      const touch = (id, extra) => {
        const c = clients.get(id) ?? { id };
        Object.assign(c, extra, { lastSeen: Date.now() });
        clients.set(id, c);
        return c;
      };

      server.ws.on('remote:hello', (data) => {
        touch(data.id, { ua: data.ua, url: data.url, connected: Date.now() });
        addLog(data.id, 'hello', `${data.ua} ${data.url}`);
      });
      server.ws.on('remote:log', (data) => {
        touch(data.id);
        for (const entry of data.entries) addLog(data.id, entry.level, entry.text);
      });
      server.ws.on('remote:stats', (data) => touch(data.id, { stats: data.stats, url: data.url }));
      server.ws.on('remote:result', (data) => {
        touch(data.id);
        pending.get(data.evalId)?.(data);
        pending.delete(data.evalId);
      });

      const isLocal = (req) => {
        const host = (req.headers.host ?? '').replace(/:\d+$/, '');
        const addr = req.socket.remoteAddress ?? '';
        return (host === 'localhost' || host === '127.0.0.1' || host === '[::1]') &&
          !req.headers['cf-ray'] && !req.headers['cf-connecting-ip'] &&
          (addr === '127.0.0.1' || addr === '::1' || addr === '::ffff:127.0.0.1');
      };
      const json = (res, status, body) => {
        res.statusCode = status;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(body));
      };
      const readBody = (req) => new Promise((resolve) => {
        let body = '';
        req.on('data', (chunk) => body += chunk);
        req.on('end', () => resolve(body));
      });
      // the most recently seen client whose user agent has `match` (any client without it)
      const pickClient = (match) => [...clients.values()]
        .filter((c) => !match || (c.ua ?? '').toLowerCase().includes(match.toLowerCase()))
        .sort((a, b) => b.lastSeen - a.lastSeen)[0];

      server.middlewares.use('/__remote', async (req, res, next) => {
        if (!isLocal(req)) return next();
        const url = new URL(req.url ?? '/', 'http://localhost');
        const route = url.pathname;

        if (route === '/clients') {
          const now = Date.now();
          return json(res, 200, [...clients.values()].map((c) => ({ ...c, secondsAgo: (now - c.lastSeen) / 1000 })));
        }
        if (route === '/log') {
          const since = parseInt(url.searchParams.get('since') ?? '0') || 0;
          const client = url.searchParams.get('client');
          return json(res, 200, { seq, logs: logs.filter((l) => l.seq > since && (!client || l.client === client)) });
        }
        if (route === '/clear') {
          logs.length = 0;
          return json(res, 200, { seq });
        }
        if (route === '/eval' && req.method === 'POST') {
          const code = await readBody(req);
          const target = pickClient(url.searchParams.get('ua'));
          if (!target) return json(res, 404, { error: 'no client: open the page with ?remote on the device' });
          const timeout = parseFloat(url.searchParams.get('timeout') ?? '60') * 1000;
          const id = ++evalId;
          const result = new Promise((resolve) => {
            pending.set(id, resolve);
            setTimeout(() => {
              if (pending.delete(id)) resolve({ error: `timeout after ${timeout / 1000} s (page frozen or reloaded?)` });
            }, timeout);
          });
          server.ws.send('remote:eval', { target: target.id, evalId: id, code });
          return json(res, 200, { client: target.id, ua: target.ua, ...(await result) });
        }
        return json(res, 404, { error: 'routes: /clients /log?since=&client= /clear POST /eval?ua=&timeout=' });
      });
    },
  };
}
