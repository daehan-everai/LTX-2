import { db } from '@/db';
import { settings } from '@/db/schema';
import { eq } from 'drizzle-orm';

export function getSetting(key: string): string | null {
  const row = db.select().from(settings).where(eq(settings.key, key)).get();
  return row?.value ?? null;
}

export function setSetting(key: string, value: string): void {
  db.insert(settings).values({ key, value }).onConflictDoUpdate({ target: settings.key, set: { value } }).run();
}

export function upsertSettings(pairs: Record<string, string>): void {
  for (const [key, value] of Object.entries(pairs)) {
    setSetting(key, value);
  }
}
