import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { getWorkerDb, getSettingSync, nowIso, markJobFinished, type JobRow } from './db';

export async function startJob(job: JobRow): Promise<number> {
  const config = JSON.parse(job.config);

  const logFile = job.log_file || path.join(process.cwd(), 'data', 'logs', `job_${job.id}.log`);
  fs.mkdirSync(path.dirname(logFile), { recursive: true });

  if (!job.log_file) {
    const db = getWorkerDb();
    db.prepare('UPDATE jobs SET log_file = ? WHERE id = ?').run(logFile, job.id);
  }

  const logFd = fs.openSync(logFile, 'a');

  let command: string;
  let args: string[];
  let cwd: string;
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    TERM: 'xterm-256color',
    FORCE_COLOR: '1',
    COLORTERM: 'truecolor',
  };

  const scriptsDir = getSettingSync('scriptsDir') || (config.scriptsDir as string) || '';

  if (job.type === 'preprocess') {
    command = 'python3';
    args = buildPreprocessArgs(config, scriptsDir);
    cwd = scriptsDir || process.cwd();
    env.PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True';
  } else if (job.type === 'training') {
    const gpuMode = (config.gpuMode as string) || 'single';
    const gpuIds = (config.gpuIds as string) || '0';
    env.CUDA_VISIBLE_DEVICES = gpuIds;
    env.PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True';
    cwd = path.dirname(scriptsDir) || process.cwd();

    if (gpuMode === 'ddp') {
      command = 'accelerate';
      args = [
        'launch',
        '--config_file',
        path.join(cwd, 'configs', 'accelerate', 'ddp.yaml'),
        path.join(scriptsDir, 'train.py'),
        config.configPath as string,
      ];
    } else {
      command = 'python3';
      args = [path.join(scriptsDir, 'train.py'), config.configPath as string];
    }
  } else if (job.type === 'merge') {
    return handleMergeJob(job.id, config, logFd);
  } else {
    fs.closeSync(logFd);
    throw new Error(`Unknown job type: ${job.type}`);
  }

  const exitCodeFile = logFile.replace(/\.log$/, '.exitcode');

  const header = `[${nowIso()}] Starting: ${command} ${args.join(' ')}\n`;
  fs.writeSync(logFd, header);

  // Wrap command in a shell that:
  //  1. Writes the real child exit code to a .exitcode file before exiting.
  //     This is the fallback so if the Node worker restarts while a detached
  //     job is running, the next worker poll can still recover the result.
  //     Without it, a successful job is wrongly marked as "failed".
  //  2. Forwards SIGINT / SIGTERM received by the shell to the child process,
  //     so the existing stop API (which sends SIGINT to the stored PID, now
  //     the shell PID) still cancels training cleanly.
  const shellEscape = (s: string) => `'${s.replace(/'/g, "'\\''")}'`;
  const innerCmd = [command, ...args].map(shellEscape).join(' ');
  const wrappedCmd = [
    `__ltx_exitcode_file=${shellEscape(exitCodeFile)}`,
    `${innerCmd} &`,
    `__ltx_child=$!`,
    `trap 'kill -INT "$__ltx_child" 2>/dev/null' INT`,
    `trap 'kill -TERM "$__ltx_child" 2>/dev/null' TERM`,
    `wait "$__ltx_child"`,
    `__ltx_ec=$?`,
    // wait can return early when interrupted by a signal; loop until child is really gone
    `while kill -0 "$__ltx_child" 2>/dev/null; do wait "$__ltx_child"; __ltx_ec=$?; done`,
    `printf '%s' "$__ltx_ec" > "$__ltx_exitcode_file"`,
    `exit "$__ltx_ec"`,
  ].join('\n');

  const child = spawn('sh', ['-c', wrappedCmd], {
    cwd,
    env,
    detached: true,
    stdio: ['ignore', logFd, logFd],
  });

  if (!child.pid) {
    fs.closeSync(logFd);
    throw new Error('Failed to spawn process');
  }

  const pidFile = logFile.replace(/\.log$/, '.pid');
  fs.writeFileSync(pidFile, String(child.pid));

  const precomputedSrc =
    job.type === 'preprocess' && config.folderPath ? path.join(config.folderPath as string, '.precomputed') : null;
  const precomputedDest =
    job.type === 'preprocess' && config.outputFolderPath
      ? path.join(config.outputFolderPath as string, '.precomputed')
      : null;

  const jobId = job.id;
  child.on('close', code => {
    const footer = `\n[${nowIso()}] Process exited with code ${code ?? 'null'}\n`;
    try {
      fs.writeSync(logFd, footer);
    } catch {
      /* fd may already be closed */
    }

    if (code === 0 && precomputedSrc && precomputedDest) {
      try {
        if (fs.existsSync(precomputedSrc)) {
          fs.mkdirSync(path.dirname(precomputedDest), { recursive: true });
          fs.renameSync(precomputedSrc, precomputedDest);
          try {
            fs.writeSync(logFd, `[${nowIso()}] Moved .precomputed → ${precomputedDest}\n`);
          } catch {
            /* log fd may already be closed on retry */
          }
        }
      } catch (err) {
        try {
          fs.writeSync(logFd, `[${nowIso()}] WARNING: could not move .precomputed: ${err}\n`);
        } catch {
          /* ignore */
        }
      }
    }

    fs.closeSync(logFd);

    markJobFinished(getWorkerDb(), jobId, code);
  });

  child.unref();

  return child.pid;
}

function buildPreprocessArgs(config: Record<string, unknown>, scriptsDir: string): string[] {
  const script = path.join(scriptsDir, 'process_dataset.py');
  const args = [script];

  const datasetPath = (config.datasetPath as string) || '';
  args.push(datasetPath);

  if (config.resolutionBuckets) {
    args.push('--resolution-buckets', config.resolutionBuckets as string);
  }

  const modelPath = (config.modelPath as string) || getSettingSync('modelPath');
  if (modelPath) args.push('--model-path', modelPath);

  const textEncoderPath = (config.textEncoderPath as string) || getSettingSync('textEncoderPath');
  if (textEncoderPath) args.push('--text-encoder-path', textEncoderPath);

  if (config.hFlip) args.push('--with-h-flip');
  if (config.withAudio) args.push('--with-audio');
  if (config.frameSampling) args.push('--frame-sampling', config.frameSampling as string);

  return args;
}

function handleMergeJob(jobId: number, config: Record<string, unknown>, logFd: number): number {
  const db = getWorkerDb();
  const sourceDirs = (config.sourceDirs as string[]) || [];
  const destDir = (config.destDir as string) || '';

  fs.writeSync(logFd, `[${nowIso()}] Merging ${sourceDirs.length} bucket(s) into ${destDir}\n`);

  try {
    for (const src of sourceDirs) {
      const pre = path.join(src, '.precomputed');
      const tag = path.basename(src);

      for (const subdir of ['latents', 'latents_h_flip', 'conditions', 'audio_latents']) {
        const srcDir = path.join(pre, subdir);
        if (!fs.existsSync(srcDir)) continue;

        const dest = path.join(destDir, '.precomputed', subdir, tag);
        fs.mkdirSync(dest, { recursive: true });
        hardLinkRecursive(srcDir, dest);
        fs.writeSync(logFd, `  Linked ${subdir}/${tag}\n`);
      }
    }

    fs.writeSync(logFd, `[${nowIso()}] Merge complete\n`);

    db.prepare("UPDATE jobs SET status = 'completed', progress = 100, completed_at = ? WHERE id = ?").run(
      nowIso(),
      jobId,
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    fs.writeSync(logFd, `[${nowIso()}] ERROR: ${msg}\n`);

    db.prepare("UPDATE jobs SET status = 'failed', error = ?, completed_at = ? WHERE id = ?").run(msg, nowIso(), jobId);
  }

  fs.closeSync(logFd);
  return -1;
}

function hardLinkRecursive(src: string, dest: string) {
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      fs.mkdirSync(destPath, { recursive: true });
      hardLinkRecursive(srcPath, destPath);
    } else if (!fs.existsSync(destPath)) {
      fs.linkSync(srcPath, destPath);
    }
  }
}
