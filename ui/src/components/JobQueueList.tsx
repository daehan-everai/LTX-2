'use client';

import type { ProcessingJob } from '@/lib/types';
import { formatDuration } from '@/lib/format';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { StatusBadge } from '@/components/StatusBadge';
import { useWorkerStatus } from '@/hooks/useWorkerStatus';
import { EmptyState } from '@/components/EmptyState';
import { X, RotateCcw, GraduationCap, Layers, Package, AlertTriangle } from 'lucide-react';

interface Props {
  jobs: ProcessingJob[];
  onCancel?: (id: number) => void;
  onReuse?: (job: ProcessingJob) => void;
}

const TYPE_ICONS = {
  preprocess: Layers,
  merge: Package,
  training: GraduationCap,
} as const;

export function JobQueueList({ jobs, onCancel, onReuse }: Props) {
  const { alive } = useWorkerStatus();
  const hasQueuedJobs = jobs.some(j => j.status === 'queued' || j.status === 'running');

  if (jobs.length === 0) {
    return <EmptyState className="surface-neo-inset rounded-2xl">No jobs in queue</EmptyState>;
  }

  return (
    <div className="space-y-1.5">
      {!alive && hasQueuedJobs && (
        <div className="flex items-center gap-2 rounded-xl border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-400">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>
            Worker process is not running — queued jobs will not start. Run{' '}
            <code className="rounded bg-yellow-500/10 px-1 font-mono text-xs">npm run worker</code> or restart with{' '}
            <code className="rounded bg-yellow-500/10 px-1 font-mono text-xs">npm run dev</code>.
          </span>
        </div>
      )}
      {jobs.map(job => {
        const TypeIcon = TYPE_ICONS[job.type] || Layers;

        return (
          <div key={job.id} className="surface-neo border-border flex items-center gap-3 rounded-xl border px-3 py-2">
            <TypeIcon className="text-muted-foreground h-4 w-4 shrink-0" />

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate font-mono text-sm font-medium">
                  {job.type === 'preprocess'
                    ? (((job.config as Record<string, unknown>).outputFolderPath as string | undefined) ?? job.name)
                    : job.name}
                </span>
                <StatusBadge status={job.status} className="shrink-0" />
              </div>
              {job.startedAt && (
                <span className="text-muted-foreground text-[10px]">
                  {formatDuration(job.startedAt, job.completedAt)}
                  {job.error && ` — ${job.error}`}
                </span>
              )}
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

            {onReuse && job.type === 'training' && ['completed', 'failed', 'cancelled'].includes(job.status) && (
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-foreground h-6 w-6 shrink-0 p-0"
                title="Reuse config"
                onClick={() => onReuse(job)}
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            )}

            {(job.status === 'queued' || job.status === 'running') && onCancel && (
              <ConfirmDialog
                title={job.status === 'running' ? 'Stop this job?' : 'Cancel this job?'}
                description={`This will ${job.status === 'running' ? 'terminate' : 'remove'} "${job.name}"${job.status === 'running' ? '. Any unsaved progress will be lost.' : ' from the queue.'}`}
                confirmLabel={job.status === 'running' ? 'Stop' : 'Cancel Job'}
                onConfirm={() => onCancel(job.id)}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-destructive h-6 w-6 shrink-0 p-0"
                  title={job.status === 'running' ? 'Stop' : 'Cancel'}
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              </ConfirmDialog>
            )}
          </div>
        );
      })}
    </div>
  );
}
