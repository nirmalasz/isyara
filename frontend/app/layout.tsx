import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ISYARA Learning",
  description: "Guided BISINDO learning path powered by ISYARA.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
