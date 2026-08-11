import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        isyara: {
          soft: "#e78197",
          primary: "#bd4e68",
          strong: "#9f3e55",
          secondary: "#6d3b3a",
          dark: "#3f2527",
          background: "#fff8f5",
          tint: "#fde8ee",
        },
      },
      borderRadius: {
        card: "8px",
      },
      fontFamily: {
        sans: ["Poppins", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
