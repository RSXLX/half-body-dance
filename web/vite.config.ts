import { defineConfig } from 'vite';
import type { Plugin, ViteDevServer } from 'vite';
import { createReadStream, existsSync, statSync, type Stats } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname, extname, resolve, sep } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const wudaoRoot = resolve(repoRoot, 'wudao');

function contentTypeFor(pathname: string): string {
  switch (extname(pathname).toLowerCase()) {
    case '.json':
      return 'application/json; charset=utf-8';
    case '.mp4':
      return 'video/mp4';
    case '.wav':
      return 'audio/wav';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.png':
      return 'image/png';
    default:
      return 'application/octet-stream';
  }
}

function wudaoStaticPlugin(): Plugin {
  return {
    name: 'wudao-static',
    configureServer(server: ViteDevServer) {
      server.middlewares.use((req: IncomingMessage, res: ServerResponse, next: () => void) => {
        if (!req.url) {
          next();
          return;
        }
        const url = new URL(req.url, 'http://localhost');
        if (!url.pathname.startsWith('/wudao/')) {
          next();
          return;
        }
        let relativePath: string;
        try {
          relativePath = decodeURIComponent(url.pathname.slice('/wudao/'.length));
        } catch {
          res.statusCode = 400;
          res.end('Bad request');
          return;
        }
        const target = resolve(wudaoRoot, relativePath);
        if (target !== wudaoRoot && !target.startsWith(`${wudaoRoot}${sep}`)) {
          res.statusCode = 403;
          res.end('Forbidden');
          return;
        }
        if (!existsSync(target)) {
          res.statusCode = 404;
          res.end('Not found');
          return;
        }
        const stats = statSync(target);
        if (!stats.isFile()) {
          res.statusCode = 404;
          res.end('Not found');
          return;
        }
        res.setHeader('Content-Type', contentTypeFor(target));
        res.setHeader('Accept-Ranges', 'bytes');
        const range = req.headers.range;
        if (range) {
          const partial = parseRange(range, stats);
          if (!partial) {
            res.statusCode = 416;
            res.setHeader('Content-Range', `bytes */${stats.size}`);
            res.end();
            return;
          }
          res.statusCode = 206;
          res.setHeader('Content-Length', String(partial.end - partial.start + 1));
          res.setHeader('Content-Range', `bytes ${partial.start}-${partial.end}/${stats.size}`);
          if (req.method === 'HEAD') {
            res.end();
            return;
          }
          createReadStream(target, partial).pipe(res);
          return;
        }
        res.setHeader('Content-Length', String(stats.size));
        if (req.method === 'HEAD') {
          res.statusCode = 200;
          res.end();
          return;
        }
        createReadStream(target).pipe(res);
      });
    },
  };
}

function parseRange(range: string, stats: Stats): { start: number; end: number } | null {
  const match = range.match(/^bytes=(\d*)-(\d*)$/);
  if (!match) return null;
  let start = match[1] ? Number(match[1]) : 0;
  let end = match[2] ? Number(match[2]) : stats.size - 1;
  if (!match[1] && match[2]) {
    const suffixLength = Number(match[2]);
    start = Math.max(0, stats.size - suffixLength);
    end = stats.size - 1;
  }
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || start >= stats.size) {
    return null;
  }
  return { start, end: Math.min(end, stats.size - 1) };
}

// Vite dev server for the new web/ workspace.
// /api/* requests are proxied to the Python dev_server.py (default 4173) so the
// existing extract-pose pipeline keeps working during Phase 1 migration.
export default defineConfig({
  root: here,
  plugins: [wudaoStaticPlugin()],
  server: {
    port: 4300,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:4173',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2020',
    sourcemap: true,
  },
});
