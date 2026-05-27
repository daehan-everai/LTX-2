import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { initSchema } from '@/db/initSchema';

const DB_DIR = path.join(process.cwd(), 'data');
const DB_PATH = path.join(DB_DIR, 'ltx-ui.db');

export interface JobRow {
  id: number;
  type: string;
  name: string;
  status: string;
  config: string;
  queue_position: number;
  pid: number | null;
  log_file: string | null;
  stop_requested: number;
  progress: number | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

let _db: Database.Database | null = null;

export function getWorkerDb(): Database.Database {
  if (_db) return _db;
  fs.mkdirSync(DB_DIR, { recursive: true });
  _db = new Database(DB_PATH);
  _db.pragma('journal_mode = WAL');
  _db.pragma('busy_timeout = 5000');
  initSchema(_db);
  return _db;
}

export function getSettingSync(key: string): string {
  const row = getWorkerDb().prepare('SELECT value FROM settings WHERE key = ?').get(key) as
    | { value: string }
    | undefined;
  return row?.value || '';
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function markJobFinished(db: Database.Database, jobId: number, exitCode: number | null): void {
  const success = exitCode === 0;
  const status = success ? 'completed' : 'failed';
  const error = !success
    ? exitCode === null
      ? 'Process exited unexpectedly'
      : `Process exited with code ${exitCode}`
    : null;
  db.prepare(
    "UPDATE jobs SET status = ?, progress = ?, error = ?, completed_at = ? WHERE id = ? AND status = 'running'",
  ).run(status, success ? 100 : null, error, nowIso(), jobId);
}
