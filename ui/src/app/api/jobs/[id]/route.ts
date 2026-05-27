import { NextResponse } from 'next/server';
import { db } from '@/db';
import { jobs } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { parseJobConfig, safeId } from '@/lib/utils';

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: idStr } = await params;
  const id = safeId(idStr);
  if (!id) {
    return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
  }

  const job = db.select().from(jobs).where(eq(jobs.id, id)).get();
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  return NextResponse.json({
    ...job,
    config: parseJobConfig(job.config) ?? {},
  });
}
