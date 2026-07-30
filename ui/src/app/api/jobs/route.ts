import { NextResponse } from 'next/server';
import { db } from '@/db';
import { jobs, sourceFolders } from '@/db/schema';
import { nextQueuePosition } from '@/db/queries';
import { eq, desc, and, type SQL } from 'drizzle-orm';
import { parseJobConfig } from '@/lib/utils';
import { execFile } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execFileAsync = promisify(execFile);

async function scanBuckets(folderPath: string, resolution: number): Promise<Array<{ key: string; fileCount: number }>> {
  const bucketScript = path.resolve(/* turbopackIgnore: true */ process.cwd(), 'scripts', 'bucket_sizes.py');

  const { stdout } = await execFileAsync('python3', [bucketScript, folderPath, '--resolution', String(resolution)], {
    timeout: 60_000,
  });

  const result = JSON.parse(stdout) as {
    resolutions: Record<string, Array<{ key: string; fileCount: number }>>;
  };

  return result.resolutions?.[String(resolution)] || [];
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const status = searchParams.get('status');
  const type = searchParams.get('type');

  type JobStatus = (typeof jobs.status.enumValues)[number];
  type JobType = (typeof jobs.type.enumValues)[number];

  const conditions: SQL[] = [];
  if (status) conditions.push(eq(jobs.status, status as JobStatus));
  if (type) conditions.push(eq(jobs.type, type as JobType));

  let query = db.select().from(jobs).orderBy(desc(jobs.createdAt));
  if (conditions.length > 0) {
    query = query.where(and(...conditions)) as typeof query;
  }
  const rows = query.all();

  return NextResponse.json(
    rows.map(r => {
      const config = parseJobConfig(r.config) ?? {};
      let outputExists: boolean | undefined;
      if (r.type === 'preprocess' && r.status === 'completed') {
        const outputFolderPath = (config as Record<string, unknown>).outputFolderPath as string | undefined;
        outputExists = outputFolderPath ? fs.existsSync(path.join(outputFolderPath, '.precomputed')) : false;
      }
      return {
        ...r,
        config,
        ...(outputExists !== undefined ? { outputExists } : {}),
      };
    }),
  );
}

export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  const config: Record<string, unknown> = { ...(body.config as Record<string, unknown>) };

  if (body.type === 'preprocess' && config.folderId) {
    const folder = db
      .select()
      .from(sourceFolders)
      .where(eq(sourceFolders.id, config.folderId as number))
      .get();

    if (folder) {
      const datasetFilename = (config.datasetFilename as string) || 'dataset.json';
      config.datasetPath = path.join(folder.path, datasetFilename);

      if (config.audioOnly) {
        // Audio-only preprocessing: flat output, no resolution/frame dimension.
        config.outputFolderPath = path.join(folder.path, '_buckets', 'audio_only');
      } else {
        const resolution = config.resolution as number;
        const frameCounts = (config.frameCounts as number[]) || [];

        const frameCount = frameCounts[0];
        if (resolution && frameCount !== undefined) {
          config.outputFolderPath = path.join(folder.path, '_buckets', `${resolution}_${frameCount}`);
        }

        if (resolution && frameCounts.length > 0) {
          try {
            const buckets = await scanBuckets(folder.path, resolution);

            const bucketStrings: string[] = [];
            for (const bucket of buckets) {
              for (const fc of frameCounts) {
                bucketStrings.push(`${bucket.key}x${fc}`);
              }
            }

            if (bucketStrings.length > 0) {
              config.resolutionBuckets = bucketStrings.join(';');
            }
          } catch (err) {
            console.error('Bucket scan failed, falling back to square:', err);
          }

          if (!config.resolutionBuckets) {
            config.resolutionBuckets = frameCounts.map(fc => `${resolution}x${resolution}x${fc}`).join(';');
          }
        }
      }
    }
  }

  const result = db
    .insert(jobs)
    .values({
      type: body.type as 'preprocess' | 'merge' | 'training',
      name: body.name as string,
      status: 'queued',
      config: JSON.stringify(config),
      queuePosition: nextQueuePosition(),
    })
    .returning()
    .get();

  return NextResponse.json({ ...result, config: parseJobConfig(result.config) ?? {} }, { status: 201 });
}
