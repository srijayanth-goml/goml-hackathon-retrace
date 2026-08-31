/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Module 6 (App UI) -- see ../../plan.md's "Module 6 -- App UI -- detailed plan" for
// why this stack (React + TS + Vite + Tailwind, no router, no state library) was
// chosen. Dev server stays on Vite's default port 5173 deliberately --
// ../backend/config.py's CORS_ALLOW_ORIGINS was already written with that exact
// origin in mind before this module existed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
  },
});
