import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ForeshadowNBA - NBA Game Prediction Dashboard",
  description: "Real-time NBA game prediction analytics powered by machine learning. Track prediction accuracy, view confusion matrices, and analyze game outcomes.",
  keywords: ["NBA", "predictions", "machine learning", "basketball", "analytics"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
