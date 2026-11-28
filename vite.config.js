import {defineConfig} from 'vite';
import path from 'path';
import mkcert from 'vite-plugin-mkcert';
import remoteDebug from './tools/remoteDebugPlugin.js';

export default defineConfig({
    base: './',
    resolve: {
        alias: {
            'webGPU-framework': path.resolve(__dirname, './src/lib/'),
        },
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        minify: true,
        sourcemap: false,
    },
    server: {
        https: true,
        // tools/remoteDebugPlugin.js through a cloudflared quick tunnel
        allowedHosts: ['.trycloudflare.com'],
    },
    plugins: [mkcert(), remoteDebug()],
});
