import { NextResponse } from 'next/server';
import { db } from '@/db';
import { settings } from '@/db/schema';
import { upsertSettings } from '@/lib/settings';

const DEFAULT_SETTINGS: Record<string, string> = {
  modelPath: '',
  textEncoderPath: '',
  outputDir: '',
  datasetDir: '',
  scriptsDir: '',
};

export async function GET() {
  const rows = db.select().from(settings).all();
  const result: Record<string, string> = { ...DEFAULT_SETTINGS };
  for (const row of rows) {
    result[row.key] = row.value;
  }
  return NextResponse.json(result);
}

export async function PUT(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  const pairs: Record<string, string> = {};
  for (const [key, value] of Object.entries(body)) {
    if (typeof value === 'string') {
      pairs[key] = value;
    }
  }
  upsertSettings(pairs);
  return NextResponse.json({ ok: true });
}
