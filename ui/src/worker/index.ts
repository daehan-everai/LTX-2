import { processQueue } from './processQueue';
import { getWorkerDb, getSettingSync } from './db';

console.log('[worker] LTX-UI job queue worker started');

const POLL_INTERVAL_MS = 1000;
const TB_SETTING_KEYS = ['tbPid', 'tbPort', 'tbLogDir', 'tbPathPrefix'] as const;

function writeHeartbeat() {
  const db = getWorkerDb();
  db.prepare(
    "INSERT INTO settings (key, value) VALUES ('worker_heartbeat', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
  ).run(new Date().toISOString());
}

function killTensorBoard() {
  const pidStr = getSettingSync('tbPid');
  if (pidStr) {
    const pid = parseInt(pidStr, 10);
    if (!isNaN(pid) && pid > 0) {
      try {
        process.kill(pid, 'SIGTERM');
        console.log(`[worker] Sent SIGTERM to TensorBoard (pid ${pid})`);
      } catch {
        /* already dead */
      }
    }
  }
  const db = getWorkerDb();
  const clearStmt = db.prepare("UPDATE settings SET value = '' WHERE key = ?");
  for (const key of TB_SETTING_KEYS) {
    clearStmt.run(key);
  }
}

function shutdown(signal: string) {
  console.log(`[worker] Received ${signal}, shutting down`);
  try {
    killTensorBoard();
  } catch (err) {
    console.error('[worker] Error killing TensorBoard:', err);
  }
  process.exit(0);
}

async function tick() {
  try {
    writeHeartbeat();
    await processQueue();
  } catch (err) {
    console.error('[worker] processQueue error:', err);
  }
}

async function schedule(): Promise<void> {
  await tick();
  setTimeout(schedule, POLL_INTERVAL_MS);
}
schedule();

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
