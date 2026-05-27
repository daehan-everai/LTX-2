// Tiny standalone process supervisor for queued jobs.
//
// Forwards SIGINT/SIGTERM to the child's process group and records the
// child's exit code to a file so the worker can recover orphaned jobs.

import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { constants } from 'node:os';

const [exitCodeFile, command, ...args] = process.argv.slice(2);

if (!exitCodeFile || !command) {
  process.exit(64);
}

function writeExitCode(code) {
  try {
    writeFileSync(exitCodeFile, String(code));
  } catch {
    /* recovery path will treat missing file as unknown */
  }
}

// `detached: true` gives the child its own process group so we can signal
// the whole group at once.
const child = spawn(command, args, { stdio: 'inherit', detached: true });

if (!child.pid) {
  writeExitCode(127);
  process.exit(127);
}

const pgid = child.pid;
let forwarded = false;

function forward(signal) {
  if (forwarded) return;
  forwarded = true;
  try {
    process.kill(-pgid, signal);
  } catch {
    /* group already gone */
  }
}

process.on('SIGINT', () => forward('SIGINT'));
process.on('SIGTERM', () => forward('SIGTERM'));
process.on('SIGHUP', () => forward('SIGHUP'));

child.on('exit', (code, signal) => {
  const finalCode =
    code !== null ? code : signal && constants.signals[signal] ? 128 + constants.signals[signal] : 1;
  writeExitCode(finalCode);
  process.exit(finalCode);
});
