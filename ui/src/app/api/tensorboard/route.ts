import { NextResponse } from 'next/server';
import { getSetting, setSetting } from '@/lib/settings';
import { toErrorMessage } from '@/lib/utils';
import { spawn } from 'child_process';

// Defaults must match run.sh and ui/next.config.ts.
const TB_HOST = process.env.TENSORBOARD_HOST || '127.0.0.1';
const TB_PORT = process.env.TENSORBOARD_PORT || '6006';
const TB_PATH_PREFIX = process.env.TENSORBOARD_PATH_PREFIX || '/tensorboard';

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function clearTbState(): void {
  setSetting('tbPid', '');
  setSetting('tbPort', '');
  setSetting('tbLogDir', '');
  setSetting('tbPathPrefix', '');
}

function killTbProcess(): void {
  const pidStr = getSetting('tbPid');
  if (pidStr) {
    const pid = parseInt(pidStr, 10);
    if (!isNaN(pid) && pid > 0 && isProcessAlive(pid)) {
      try {
        process.kill(pid, 'SIGTERM');
      } catch {
        /* already dead */
      }
    }
  }
  clearTbState();
}

export async function GET() {
  const pidStr = getSetting('tbPid');
  const pid = pidStr ? parseInt(pidStr, 10) : 0;
  const port = parseInt(getSetting('tbPort') || '0', 10);
  const logDir = getSetting('tbLogDir') || null;
  const storedPrefix = getSetting('tbPathPrefix') || '';

  if (!pid) {
    return NextResponse.json({ running: false, port: 0, logDir: null });
  }

  if (!isProcessAlive(pid)) {
    clearTbState();
    return NextResponse.json({ running: false, port: 0, logDir: null });
  }

  // Kill stale TB processes that were spawned without the current path_prefix
  // (e.g. before the reverse-proxy migration). The hook will respawn a fresh one.
  if (storedPrefix !== TB_PATH_PREFIX) {
    killTbProcess();
    return NextResponse.json({ running: false, port: 0, logDir: null });
  }

  return NextResponse.json({ running: true, port, logDir });
}

export async function POST(req: Request) {
  const body = await req.json();
  const logDir: string = body.logDir;

  if (!logDir) {
    return NextResponse.json({ error: 'logDir is required' }, { status: 400 });
  }

  killTbProcess();

  try {
    const child = spawn(
      'tensorboard',
      [
        '--logdir',
        logDir,
        '--port',
        String(TB_PORT),
        '--host',
        TB_HOST,
        '--path_prefix',
        TB_PATH_PREFIX,
        '--reload_interval',
        '5',
      ],
      {
        detached: true,
        stdio: 'ignore',
      },
    );

    if (!child.pid) {
      return NextResponse.json({ error: 'Failed to spawn tensorboard process' }, { status: 500 });
    }

    child.unref();

    setSetting('tbPid', String(child.pid));
    setSetting('tbPort', TB_PORT);
    setSetting('tbLogDir', logDir);
    setSetting('tbPathPrefix', TB_PATH_PREFIX);

    return NextResponse.json({ running: true, port: parseInt(TB_PORT, 10), logDir });
  } catch (err: unknown) {
    return NextResponse.json({ error: toErrorMessage(err) }, { status: 500 });
  }
}

export async function DELETE() {
  killTbProcess();
  return NextResponse.json({ running: false });
}
