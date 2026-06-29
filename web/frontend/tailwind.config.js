/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#09090B",
        panel: "#18181B",
        panel2: "#202024",
        edge: "#2a2a30",
        up: "#22C55E",
        down: "#EF4444",
        warn: "#F97316",
        info: "#3B82F6",
        accent: "#8B5CF6",
        muted: "#71717A",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
