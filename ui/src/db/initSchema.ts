import type Database from 'better-sqlite3';

const SCHEMA_SQL = `
  CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS source_folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    media_type TEXT NOT NULL DEFAULT 'videos',
    file_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    config TEXT NOT NULL,
    queue_position INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    log_file TEXT,
    stop_requested INTEGER NOT NULL DEFAULT 0,
    progress INTEGER,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS training_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    buckets TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
`;

export function initSchema(sqlite: Database.Database): void {
  sqlite.exec(SCHEMA_SQL);
}
