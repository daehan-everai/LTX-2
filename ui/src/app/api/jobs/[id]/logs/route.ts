import { NextResponse } from 'next/server';
import { db } from '@/db';
import { jobs } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { safeId } from '@/lib/utils';
import { getSetting } from '@/lib/settings';
import fs from 'fs';

const WORKER_STALE_MS = 6000;
const HEAD_BYTES = 256 * 1024;
const TAIL_BYTES = 512 * 1024;
const THRESHOLD = HEAD_BYTES + TAIL_BYTES;
const MAX_CHUNK = 64 * 1024;

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: idStr } = await params;
  const id = safeId(idStr);
  if (!id) {
    return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
  }

  const job = db.select().from(jobs).where(eq(jobs.id, id)).get();
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!job.logFile || !fs.existsSync(job.logFile)) {
    return NextResponse.json({ error: 'No log file' }, { status: 404 });
  }

  const logFile = job.logFile;
  const encoder = new TextEncoder();
  let offset = 0;
  let closed = false;
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let doneInterval: ReturnType<typeof setInterval> | null = null;

  const cleanup = () => {
    closed = true;
    if (pollInterval) clearInterval(pollInterval);
    if (doneInterval) clearInterval(doneInterval);
    pollInterval = null;
    doneInterval = null;
  };

  const stream = new ReadableStream({
    start(controller) {
      try {
        const initialStat = fs.statSync(logFile);
        if (initialStat.size > THRESHOLD) {
          const fd = fs.openSync(logFile, 'r');
          const headBuf = Buffer.alloc(HEAD_BYTES);
          const headRead = fs.readSync(fd, headBuf, 0, HEAD_BYTES, 0);
          fs.closeSync(fd);

          let headEnd = headRead;
          while (headEnd > 0 && headBuf[headEnd - 1] !== 0x0a) headEnd--;
          if (headEnd === 0) headEnd = headRead;

          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(headBuf.slice(0, headEnd).toString('utf-8'))}\n\n`),
          );

          const rawTailStart = initialStat.size - TAIL_BYTES;
          const fd2 = fs.openSync(logFile, 'r');
          const alignBuf = Buffer.alloc(256);
          const alignRead = fs.readSync(fd2, alignBuf, 0, 256, rawTailStart);
          fs.closeSync(fd2);
          const nlIdx = alignBuf.indexOf(0x0a, 0);
          offset = rawTailStart + (nlIdx !== -1 && nlIdx < alignRead ? nlIdx + 1 : 0);

          const skipped = offset - headEnd;
          controller.enqueue(encoder.encode(`data: ${JSON.stringify('__GAP:' + skipped)}\n\n`));
        }
      } catch {
        // Fall through to normal full-file streaming if anything goes wrong
      }

      const sendChunk = () => {
        if (closed) return;
        try {
          const stat = fs.statSync(logFile);
          if (stat.size > offset) {
            const toRead = Math.min(stat.size - offset, MAX_CHUNK);
            const fd = fs.openSync(logFile, 'r');
            const buf = Buffer.alloc(toRead);
            fs.readSync(fd, buf, 0, toRead, offset);
            fs.closeSync(fd);
            offset += toRead;
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(buf.toString('utf-8'))}\n\n`));
          }
        } catch {
          // File may have been removed
        }
      };

      sendChunk();
      pollInterval = setInterval(sendChunk, 200);

      doneInterval = setInterval(() => {
        const currentJob = db.select().from(jobs).where(eq(jobs.id, id)).get();
        if (!currentJob || ['completed', 'failed', 'cancelled'].includes(currentJob.status)) {
          sendChunk();
          controller.enqueue(encoder.encode(`data: ${JSON.stringify('__DONE__')}\n\n`));
          cleanup();
          controller.close();
          return;
        }

        if (currentJob.status === 'running') {
          const lastSeen = getSetting('worker_heartbeat');
          const stale = !lastSeen || Date.now() - new Date(lastSeen).getTime() > WORKER_STALE_MS;
          if (stale) {
            cleanup();
            try {
              controller.close();
            } catch {
              /* already closed */
            }
          }
        }
      }, 1000);

      req.signal.addEventListener(
        'abort',
        () => {
          cleanup();
          try {
            controller.close();
          } catch {
            // already closed
          }
        },
        { once: true },
      );
    },
    cancel() {
      cleanup();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
