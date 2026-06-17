/**
 * One-off migration: backfill existing job names to the unified prefixed format.
 *
 *   preprocess -> "Preprocess: {bucket path}"
 *   merge      -> "Dataset: {dest path}"
 *   training   -> "Train: {output dir}"
 *
 * Idempotent: rows already carrying the correct prefix are skipped.
 *
 * Usage (from the `ui/` directory):
 *   npx tsx scripts/migrate_job_names.ts          # apply changes
 *   npx tsx scripts/migrate_job_names.ts --dry    # preview only
 */
import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

const DB_PATH = path.join(process.cwd(), 'data', 'ltx-ui.db');
const DRY_RUN = process.argv.includes('--dry') || process.argv.includes('--dry-run');

interface JobRow {
  id: number;
  type: 'preprocess' | 'merge' | 'training';
  name: string;
  config: string;
}

function parseConfig(raw: string): Record<string, unknown> {
  try {
    return (JSON.parse(raw) as Record<string, unknown>) ?? {};
  } catch {
    return {};
  }
}

function nextName(job: JobRow): string | null {
  const cfg = parseConfig(job.config);

  switch (job.type) {
    case 'preprocess': {
      if (job.name.startsWith('Preprocess: ')) return null;
      const base = (cfg.outputFolderPath as string | undefined) ?? job.name;
      return `Preprocess: ${base}`;
    }
    case 'merge': {
      if (job.name.startsWith('Dataset: ')) return null;
      const dest = cfg.destDir as string | undefined;
      // Old name was "Build: {dataset name}". Prefer the real dest path from config.
      const base = dest ?? job.name.replace(/^Build:\s*/, '');
      return `Dataset: ${base}`;
    }
    case 'training': {
      if (job.name.startsWith('Train: ')) return null;
      const base = (cfg.outputDir as string | undefined) ?? job.name;
      return `Train: ${base}`;
    }
    default:
      return null;
  }
}

function main(): void {
  if (!fs.existsSync(DB_PATH)) {
    console.error(`Database not found at ${DB_PATH}. Run this from the ui/ directory.`);
    process.exit(1);
  }

  const sqlite = new Database(DB_PATH);
  sqlite.pragma('busy_timeout = 5000');

  const rows = sqlite.prepare('SELECT id, type, name, config FROM jobs').all() as JobRow[];
  const update = sqlite.prepare('UPDATE jobs SET name = ? WHERE id = ?');

  let changed = 0;
  let skipped = 0;

  const apply = sqlite.transaction((jobRows: JobRow[]) => {
    for (const job of jobRows) {
      const newName = nextName(job);
      if (!newName || newName === job.name) {
        skipped++;
        continue;
      }
      console.log(`#${job.id} [${job.type}]\n  - ${job.name}\n  + ${newName}`);
      if (!DRY_RUN) update.run(newName, job.id);
      changed++;
    }
  });

  apply(rows);

  console.log(
    `\n${DRY_RUN ? '[dry run] ' : ''}Done. ${changed} renamed, ${skipped} unchanged, ${rows.length} total.`,
  );
  sqlite.close();
}

main();
