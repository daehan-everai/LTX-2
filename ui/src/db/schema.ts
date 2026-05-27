import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';

export const settings = sqliteTable('settings', {
  key: text('key').primaryKey(),
  value: text('value').notNull(),
});

export const sourceFolders = sqliteTable('source_folders', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  path: text('path').notNull().unique(),
  name: text('name').notNull(),
  mediaType: text('media_type', { enum: ['images', 'videos', 'mixed'] })
    .notNull()
    .default('videos'),
  fileCount: integer('file_count').notNull().default(0),
  createdAt: text('created_at')
    .notNull()
    .$defaultFn(() => new Date().toISOString()),
});

export const jobs = sqliteTable('jobs', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  type: text('type', { enum: ['preprocess', 'merge', 'training'] }).notNull(),
  name: text('name').notNull(),
  status: text('status', { enum: ['queued', 'running', 'completed', 'failed', 'cancelled'] })
    .notNull()
    .default('queued'),
  config: text('config').notNull(),
  queuePosition: integer('queue_position').notNull().default(0),
  pid: integer('pid'),
  logFile: text('log_file'),
  stopRequested: integer('stop_requested', { mode: 'boolean' }).notNull().default(false),
  progress: integer('progress'),
  error: text('error'),
  startedAt: text('started_at'),
  completedAt: text('completed_at'),
  createdAt: text('created_at')
    .notNull()
    .$defaultFn(() => new Date().toISOString()),
});

export const trainingDatasets = sqliteTable('training_datasets', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull().unique(),
  path: text('path').notNull(),
  buckets: text('buckets').notNull(),
  createdAt: text('created_at')
    .notNull()
    .$defaultFn(() => new Date().toISOString()),
});
