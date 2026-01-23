import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Predictive Maintenance Terminal | Real-time Bearing Health Monitoring",
  description: "Real-time bearing health monitoring dashboard with predictive maintenance capabilities. Visualizes equipment degradation through vibration analysis using interactive candlestick charts.",
  keywords: ["predictive maintenance", "bearing health", "vibration analysis", "industrial IoT", "machine monitoring", "RMS", "condition monitoring"],
  authors: [{ name: "Alan Watts" }],
  openGraph: {
    title: "Predictive Maintenance Terminal",
    description: "Real-time bearing health monitoring dashboard with predictive maintenance capabilities. Visualizes equipment degradation through vibration analysis.",
    url: "https://frontend-two-ashy-yla6dtp7w4.vercel.app",
    siteName: "Predictive Maintenance Terminal",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Predictive Maintenance Dashboard showing real-time bearing vibration analysis",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Predictive Maintenance Terminal",
    description: "Real-time bearing health monitoring dashboard with predictive maintenance capabilities.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
