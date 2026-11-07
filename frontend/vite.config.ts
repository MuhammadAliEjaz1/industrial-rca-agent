import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./", // relative asset paths: works on GitHub Pages, Hugging Face static Spaces and any sub-folder
  server: { port: 5173 },
  test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"], globals: true },
});
