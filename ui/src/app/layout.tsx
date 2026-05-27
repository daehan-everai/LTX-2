import type { Metadata } from 'next';
import localFont from 'next/font/local';
import './globals.css';
import { Providers } from '@/components/Providers';
import { Sidebar } from '@/components/Sidebar';

const googleSans = localFont({
  variable: '--font-google-sans',
  display: 'swap',
  src: [
    { path: '../../public/fonts/GoogleSans-Regular.woff2', weight: '400', style: 'normal' },
    { path: '../../public/fonts/GoogleSans-Medium.woff2', weight: '500', style: 'normal' },
    { path: '../../public/fonts/GoogleSans-SemiBold.woff2', weight: '600', style: 'normal' },
    { path: '../../public/fonts/GoogleSans-Bold.woff2', weight: '700', style: 'normal' },
  ],
});

const geistMono = localFont({
  variable: '--font-geist-mono',
  display: 'swap',
  src: [
    { path: '../../public/fonts/GeistMono-Regular.woff2', weight: '400', style: 'normal' },
    { path: '../../public/fonts/GeistMono-Medium.woff2', weight: '500', style: 'normal' },
  ],
});

export const metadata: Metadata = {
  title: 'LTX Trainer',
  description: 'Training UI for LTX-2 video generation models',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${googleSans.variable} ${geistMono.variable} dark h-full antialiased`}>
      <body className="bg-background text-foreground h-full">
        <Providers>
          <div className="relative flex h-full overflow-hidden">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_78%_8%,oklch(0.7_0.19_235_/_0.18),transparent_35%),radial-gradient(circle_at_0%_0%,oklch(0.68_0.2_294_/_0.2),transparent_30%)]" />
            <Sidebar />
            <main className="relative z-10 flex-1 overflow-auto p-4 md:p-6">
              <div className="app-shell min-h-full rounded-3xl p-3 md:p-4">{children}</div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
