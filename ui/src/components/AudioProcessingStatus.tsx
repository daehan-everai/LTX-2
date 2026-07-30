'use client';

import { useMemo } from 'react';
import type { ProcessingJob } from '@/lib/types';
import { formatDuration } from '@/lib/format';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import { Music, AlertTriangle } from 'lucide-react';

interface Props {
  jobs: ProcessingJob[];
  folderId: number;
}

interface AudioJobConfig {
  folderId?: number;
  audioOnly?: boolean;
  outputFolderPath?: string;
  datasetPath?: string;
  datasetFilename?: string;
  maxDuration?: number | null;
}

export function AudioProcessingStatus({ jobs, folderId }: Props) {
  // Audio-only folders have a single flat `audio_only` bucket, so the jobs are
  // listed newest-first rather than laid out on a resolution x frames matrix.
  const audioJobs = useMemo(
    () =>
      jobs.filter(job => {
        if (job.type !== 'preprocess') return false;
        const cfg = job.config as AudioJobConfig;
        return cfg.folderId === folderId && cfg.audioOnly === true;
      }),
    [jobs, folderId],
  );

  if (audioJobs.length === 0) {
    return <EmptyState>No audio processing jobs for this folder yet. Configure and queue above.</EmptyState>;
  }

  return (
    <div className="space-y-1.5">
      {audioJobs.map(job => {
        const cfg = job.config as AudioJobConfig;
        const outputMissing = job.status === 'completed' && job.outputExists === false;

        const details = [
          // Jobs discovered on disk carry only the full dataset path.
          cfg.datasetFilename || cfg.datasetPath?.split('/').pop(),
          cfg.maxDuration != null ? `max ${cfg.maxDuration}s` : null,
          job.startedAt ? formatDuration(job.startedAt, job.completedAt) : null,
          job.error,
        ].filter(Boolean);

        return (
          <div key={job.id} className="surface-neo border-border flex items-center gap-3 rounded-xl border px-3 py-2">
            <Music className="h-4 w-4 shrink-0 text-violet-400" />

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate font-mono text-sm font-medium">{cfg.outputFolderPath || job.name}</span>
                <StatusBadge status={job.status} progress={job.progress} className="shrink-0" />
                {outputMissing && (
                  <span className="text-destructive flex shrink-0 items-center gap-1 text-[10px]">
                    <AlertTriangle className="h-3 w-3" />
                    output missing
                  </span>
                )}
              </div>
              {details.length > 0 && <span className="text-muted-foreground text-[10px]">{details.join(' — ')}</span>}
            </div>

            {job.status === 'running' && job.progress != null && (
              <div className="w-16 shrink-0">
                <div className="surface-neo-inset bg-muted h-1.5 w-full rounded-full">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <span className="text-muted-foreground text-[10px] tabular-nums">{job.progress}%</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
