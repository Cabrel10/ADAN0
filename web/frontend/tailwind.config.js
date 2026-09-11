/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0B1220",
        panel: "#111B2E",
        panel2: "#16233A",
        edge: "#243352",
        up: "#00FF88",
        down: "#FF4D4D",
        info: "#3B82F6",
        warn: "#FFC857",
        accent: "#8B5CF6",
        muted: "#64748B",
        txt: "#CBD5E1",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
