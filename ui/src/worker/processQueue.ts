import fs from 'fs';
import { getWorkerDb, nowIso, markJobFinished, type JobRow } from './db';
import { startJob } from './startJob';

function readExitCode(logFile: string | null): number | null {
  if (!logFile) return null;
  const exitCodeFile = logFile.replace(/\.log$/, '.exitcode');
  try {
    const raw = fs.readFileSync(exitCodeFile, 'utf-8').trim();
    const code = parseInt(raw, 10);
    return isNaN(code) ? null : code;
  } catch {
    return null;
  }
}

export async function processQueue(): Promise<void> {
  const sqlite = getWorkerDb();

  const running = sqlite.prepare("SELECT id FROM jobs WHERE status = 'running' LIMIT 1").get() as
    | { id: number }
    | undefined;

  if (running) {
    const job = sqlite.prepare('SELECT * FROM jobs WHERE id = ?').get(running.id) as JobRow;

    if (job.pid && job.pid > 0) {
      let processAlive = true;
      try {
        process.kill(job.pid, 0);
      } catch {
        processAlive = false;
      }

      if (!processAlive) {
        console.log(`[worker] Job ${job.id} (${job.name}) process exited`);
        markJobFinished(sqlite, job.id, readExitCode(job.log_file));
        return;
      }

      if (job.stop_requested) {
        try {
          process.kill(job.pid, 'SIGINT');
          console.log(`[worker] Sent SIGINT to job ${job.id} (pid ${job.pid})`);
        } catch {
          /* already dead */
        }
        sqlite
          .prepare("UPDATE jobs SET status = 'cancelled', completed_at = ? WHERE id = ? AND status = 'running'")
          .run(nowIso(), job.id);
      }
    }

    return;
  }

  const next = sqlite
    .prepare("SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_position ASC LIMIT 1")
    .get() as JobRow | undefined;

  if (!next) return;

  console.log(`[worker] Starting job ${next.id}: ${next.name} (${next.type})`);

  sqlite.prepare("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?").run(nowIso(), next.id);

  try {
    const pid = await startJob(next);
    sqlite.prepare('UPDATE jobs SET pid = ? WHERE id = ?').run(pid, next.id);
    console.log(`[worker] Job ${next.id} started with pid ${pid}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[worker] Failed to start job ${next.id}:`, message);
    sqlite
      .prepare("UPDATE jobs SET status = 'failed', error = ?, completed_at = ? WHERE id = ?")
      .run(message, nowIso(), next.id);
  }
}
