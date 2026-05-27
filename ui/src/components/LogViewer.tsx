'use client';

import { useEffect, useRef } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { cn } from '@/lib/utils';

interface Props {
  jobId: number | null;
  className?: string;
}

const TERMINAL_THEME = {
  background: 'rgba(0, 0, 0, 0)',
  foreground: '#e2e8f0',
  cursor: '#64748b',
  cursorAccent: '#0d1220',
  selectionBackground: '#334155',
  black: '#1e293b',
  red: '#f87171',
  green: '#34d399',
  yellow: '#fbbf24',
  blue: '#60a5fa',
  magenta: '#c084fc',
  cyan: '#22d3ee',
  white: '#e2e8f0',
  brightBlack: '#475569',
  brightRed: '#fca5a5',
  brightGreen: '#6ee7b7',
  brightYellow: '#fcd34d',
  brightBlue: '#93c5fd',
  brightMagenta: '#d8b4fe',
  brightCyan: '#67e8f9',
  brightWhite: '#f8fafc',
} as const;

type CachedTerminal = {
  jobId: number;
  term: Terminal;
  fit: FitAddon;
  wrapper: HTMLDivElement;
  es: EventSource;
};

let cached: CachedTerminal | null = null;

function disposeCache() {
  if (!cached) return;
  try {
    cached.es.close();
  } catch {
    /* ignore */
  }
  try {
    cached.term.dispose();
  } catch {
    /* ignore */
  }
  cached.wrapper.remove();
  cached = null;
}

function createCache(jobId: number, host: HTMLElement): CachedTerminal {
  const wrapper = document.createElement('div');
  wrapper.style.width = '100%';
  wrapper.style.height = '100%';

  host.appendChild(wrapper);

  const term = new Terminal({
    convertEol: true,
    cursorBlink: false,
    disableStdin: true,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    fontSize: 12,
    lineHeight: 1.25,
    scrollback: 10_000,
    theme: TERMINAL_THEME,
    allowProposedApi: true,
    allowTransparency: true,
  });

  const fit = new FitAddon();
  term.loadAddon(fit);
  term.open(wrapper);

  const es = new EventSource(`/api/jobs/${jobId}/logs`);

  es.onmessage = event => {
    let data: unknown;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }
    if (data === '__DONE__') {
      es.close();
      return;
    }
    if (typeof data === 'string' && data.startsWith('__GAP:')) {
      const skipped = parseInt(data.slice(6), 10);
      const mb = (skipped / (1024 * 1024)).toFixed(1);
      term.write(`\r\n\x1b[2m\x1b[33m${'─'.repeat(20)} ${mb} MB omitted ${'─'.repeat(20)}\x1b[0m\r\n\r\n`);
      return;
    }
    if (typeof data === 'string' && data.length > 0) {
      term.write(data);
    }
  };

  es.onerror = () => es.close();

  return { jobId, term, fit, wrapper, es };
}

export function LogViewer({ jobId, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = containerRef.current;
    if (!host || !jobId) return;

    const needsNew = !cached || cached.jobId !== jobId || cached.es.readyState === EventSource.CLOSED;

    let entry: CachedTerminal;
    if (needsNew) {
      disposeCache();
      entry = createCache(jobId, host);
      cached = entry;
    } else {
      entry = cached as CachedTerminal;
      if (entry.wrapper.parentNode !== host) {
        host.appendChild(entry.wrapper);
      }
    }

    const safeFit = () => {
      try {
        entry.fit.fit();
      } catch {
        /* container not visible yet */
      }
    };

    const raf = requestAnimationFrame(safeFit);
    const resizeObserver = new ResizeObserver(safeFit);
    resizeObserver.observe(host);

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      // Detach the wrapper but keep the Terminal + EventSource alive in the
      // module cache so a remount (e.g. navigating back to /runs) is instant.
      if (entry.wrapper.parentNode === host) {
        host.removeChild(entry.wrapper);
      }
    };
  }, [jobId]);

  return <div ref={containerRef} className={cn('p-2', className)} style={{ height: 360 }} />;
}
