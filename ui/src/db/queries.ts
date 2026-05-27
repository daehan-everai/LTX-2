import { sql, eq } from 'drizzle-orm';
import { db } from './index';
import { jobs } from './schema';

export function nextQueuePosition(): number {
  const row = db
    .select({ maxPos: sql<number | null>`MAX(${jobs.queuePosition})` })
    .from(jobs)
    .where(eq(jobs.status, 'queued'))
    .get();
  return (row?.maxPos ?? -1) + 1;
}
