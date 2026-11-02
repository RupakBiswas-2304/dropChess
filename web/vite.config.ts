import { defineConfig } from "vite";

// In dev (`npm run dev`), the Rust server must be running on :8080; Vite proxies to it.
export default defineConfig({
  server: {
    host: true,
    proxy: {
      "/ws": { target: "ws://localhost:8080", ws: true },
      "/info": "http://localhost:8080",
    },
  },
});
