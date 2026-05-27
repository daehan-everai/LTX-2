import { NextResponse } from 'next/server';
import { db } from '@/db';
import { sourceFolders, jobs } from '@/db/schema';
import { eq, like } from 'drizzle-orm';
import { safeId, parseJobConfig } from '@/lib/utils';
import fs from 'fs';
import path from 'path';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png']);
const VIDEO_EXTS = new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm']);

function scanFolderMeta(folderPath: string): {
  fileCount: number;
  mediaType: 'images' | 'videos' | 'mixed';
} {
  if (!fs.existsSync(folderPath)) {
    return { fileCount: 0, mediaType: 'videos' };
  }

  const entries = fs.readdirSync(folderPath);
  let images = 0;
  let videos = 0;

  for (const entry of entries) {
    const ext = path.extname(entry).toLowerCase();
    if (IMAGE_EXTS.has(ext)) images++;
    else if (VIDEO_EXTS.has(ext)) videos++;
  }

  const mediaType: 'images' | 'videos' | 'mixed' =
    images > 0 && videos > 0 ? 'mixed' : images > 0 ? 'images' : 'videos';

  return { fileCount: images + videos, mediaType };
}

const BUCKET_DIR_PATTERN = /^(\d+)_(\d+)$/;

function syncBucketJobs(folderId: number, folderPath: string) {
  const bucketsDir = path.join(folderPath, '_buckets');
  if (!fs.existsSync(bucketsDir)) return;

  const entries = fs.readdirSync(bucketsDir, { withFileTypes: true });
  const bucketDirs = entries.filter(e => e.isDirectory() && BUCKET_DIR_PATTERN.test(e.name));
  if (bucketDirs.length === 0) return;

  const existingNames = new Set(
    db
      .select({ name: jobs.name })
      .from(jobs)
      .where(like(jobs.name, `${folderPath}/_buckets/%`))
      .all()
      .map(r => r.name),
  );

  for (const dir of bucketDirs) {
    const match = BUCKET_DIR_PATTERN.exec(dir.name)!;
    const res = Number(match[1]);
    const fc = Number(match[2]);
    const jobName = `${folderPath}/_buckets/${res}_${fc}`;

    if (existingNames.has(jobName)) continue;

    const bucketPath = path.join(bucketsDir, dir.name);
    const precomputed = path.join(bucketPath, '.precomputed');
    const hFlip = fs.existsSync(path.join(precomputed, 'latents_h_flip'));
    const withAudio = fs.existsSync(path.join(precomputed, 'audio_latents'));
    const frameSampling = fc >= 100 ? 'uniform' : 'head';

    db.insert(jobs)
      .values({
        type: 'preprocess',
        name: jobName,
        status: 'completed',
        config: JSON.stringify({
          folderId,
          resolution: res,
          frameCounts: [fc],
          folderPath,
          outputFolderPath: bucketPath,
          datasetPath: path.join(folderPath, 'dataset.json'),
          hFlip,
          withAudio,
          frameSampling,
        }),
        queuePosition: 0,
        completedAt: new Date().toISOString(),
      })
      .run();
  }
}

export async function GET() {
  const rows = db.select().from(sourceFolders).all();
  return NextResponse.json(rows);
}

export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  const folderPath = body.path as string | undefined;

  if (!folderPath) {
    return NextResponse.json({ error: 'path is required' }, { status: 400 });
  }

  const parts = folderPath.split('/').filter(Boolean);
  const name = (body.name as string) || parts[parts.length - 1] || 'folder';

  const meta = scanFolderMeta(folderPath);

  const result = db
    .insert(sourceFolders)
    .values({
      path: folderPath,
      name,
      mediaType: meta.mediaType,
      fileCount: meta.fileCount,
    })
    .returning()
    .get();

  syncBucketJobs(result.id, folderPath);

  return NextResponse.json(result, { status: 201 });
}

const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled'] as const;

export async function PATCH(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = safeId(searchParams.get('id'));
  if (!id) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }

  const folder = db.select().from(sourceFolders).where(eq(sourceFolders.id, id)).get();
  if (!folder) {
    return NextResponse.json({ error: 'folder not found' }, { status: 404 });
  }

  const meta = scanFolderMeta(folder.path);
  db.update(sourceFolders)
    .set({ fileCount: meta.fileCount, mediaType: meta.mediaType })
    .where(eq(sourceFolders.id, id))
    .run();

  syncBucketJobs(id, folder.path);

  const candidates = db
    .select()
    .from(jobs)
    .where(like(jobs.name, `${folder.path}/_buckets/%`))
    .all()
    .filter(j => (TERMINAL_STATUSES as readonly string[]).includes(j.status));

  for (const job of candidates) {
    const cfg = parseJobConfig(job.config) as Record<string, unknown> | null;
    const outputFolderPath = cfg?.outputFolderPath as string | undefined;
    if (outputFolderPath && !fs.existsSync(outputFolderPath)) {
      db.delete(jobs).where(eq(jobs.id, job.id)).run();
    }
  }

  const updated = db.select().from(sourceFolders).where(eq(sourceFolders.id, id)).get();
  return NextResponse.json(updated);
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = safeId(searchParams.get('id'));
  if (!id) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }
  db.delete(sourceFolders).where(eq(sourceFolders.id, id)).run();
  return NextResponse.json({ ok: true });
}
