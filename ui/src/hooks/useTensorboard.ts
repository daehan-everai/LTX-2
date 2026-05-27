'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiDelete, apiFetch, apiPost } from '@/lib/api';
import type { ProcessingJob } from '@/lib/types';

interface TbStatus {
  running: boolean;
  port: number;
  logDir: string | null;
  error?: string;
}

const TB_QUERY_KEY = ['tensorboard'] as const;
const TB_READY_POLL_MS = 500;
const TB_READY_TIMEOUT_MS = 15_000;

export function useTensorboard(activeJob: ProcessingJob | null) {
  const qc = useQueryClient();
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastStartedLogDir = useRef<string | null>(null);
  const readinessTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const status = useQuery<TbStatus>({
    queryKey: TB_QUERY_KEY,
    queryFn: () => apiFetch<TbStatus>('/api/tensorboard'),
    refetchInterval: 3000,
    refetchIntervalInBackground: false,
  });

  const clearReadinessProbe = useCallback(() => {
    if (readinessTimer.current) {
      clearTimeout(readinessTimer.current);
      readinessTimer.current = null;
    }
  }, []);

  const probeReadiness = useCallback(() => {
    clearReadinessProbe();
    const deadline = Date.now() + TB_READY_TIMEOUT_MS;
    const check = async () => {
      try {
        const res = await fetch('/tensorboard/', { method: 'HEAD' });
        if (res.ok) {
          setReady(true);
          return;
        }
      } catch {
        // not ready yet
      }
      if (Date.now() < deadline) {
        readinessTimer.current = setTimeout(check, TB_READY_POLL_MS);
      } else {
        setError('TensorBoard is taking too long to respond');
      }
    };
    check();
  }, [clearReadinessProbe]);

  const running = status.data?.running ?? false;
  const port = status.data?.port ?? 0;
  const logDir = status.data?.logDir ?? null;

  // Probe readiness on running→true transitions only.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (running && !wasRunning.current) {
      setReady(false);
      probeReadiness();
    } else if (!running) {
      setReady(false);
      clearReadinessProbe();
    }
    wasRunning.current = running;
  }, [running, probeReadiness, clearReadinessProbe]);

  useEffect(() => () => clearReadinessProbe(), [clearReadinessProbe]);

  const startMut = useMutation({
    mutationFn: (dir: string) => apiPost<TbStatus>('/api/tensorboard', { logDir: dir }),
    onSuccess: data => {
      setError(data.error ?? null);
      qc.setQueryData<TbStatus>(TB_QUERY_KEY, data);
    },
    onError: err => {
      setError(err instanceof Error ? err.message : 'Failed to start');
    },
  });

  const stopMut = useMutation({
    mutationFn: () => apiDelete<TbStatus>('/api/tensorboard'),
    onSuccess: () => {
      clearReadinessProbe();
      setReady(false);
      lastStartedLogDir.current = null;
      qc.setQueryData<TbStatus>(TB_QUERY_KEY, { running: false, port: 0, logDir: null });
    },
  });

  const start = useCallback(
    async (dir: string) => {
      setError(null);
      lastStartedLogDir.current = dir;
      await startMut.mutateAsync(dir);
    },
    [startMut],
  );

  const stop = useCallback(async () => {
    await stopMut.mutateAsync();
  }, [stopMut]);

  // Auto-start/auto-stop tied to the active training job. The
  // `lastStartedLogDir` ref prevents re-firing `start` for the same logDir
  // when the status query polls back `running:true`.
  useEffect(() => {
    if (activeJob && activeJob.type === 'training' && activeJob.status === 'running') {
      const cfg = activeJob.config as Record<string, unknown>;
      const outputDir = (cfg.outputDir as string) || '';
      if (!outputDir) return;
      const desiredLogDir = `${outputDir}/tensorboard`;
      if (!running && lastStartedLogDir.current !== desiredLogDir) {
        void start(desiredLogDir);
      }
    } else if (!activeJob && running) {
      void stop();
    }
  }, [activeJob, running, start, stop]);

  return { running, ready, port, logDir, error, start, stop };
}
