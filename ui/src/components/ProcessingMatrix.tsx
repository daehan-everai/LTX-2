'use client';

import { useMemo } from 'react';
import type { ProcessingJob } from '@/lib/types';
import { RESOLUTION_OPTIONS, FRAME_COUNT_OPTIONS } from '@/lib/types';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';

interface Props {
  jobs: ProcessingJob[];
  folderId: number;
}

export function ProcessingMatrix({ jobs, folderId }: Props) {
  // Jobs arrive newest-first (DESC). Keep the first occurrence per key so the
  // latest re-queued job is shown rather than the original failed one.
  const jobMap = useMemo(() => {
    const m = new Map<string, ProcessingJob>();
    for (const job of jobs) {
      if (job.type !== 'preprocess') continue;
      const cfg = job.config as {
        folderId?: number;
        resolution?: number;
        frameCounts?: number[];
      };
      if (cfg.folderId !== folderId) continue;
      const res = cfg.resolution;
      const frames = cfg.frameCounts;
      if (res && frames) {
        for (const f of frames) {
          const key = `${res}_${f}`;
          if (!m.has(key)) m.set(key, job);
        }
      }
    }
    return m;
  }, [jobs, folderId]);

  const activeResolutions = useMemo(
    () => RESOLUTION_OPTIONS.filter(res => FRAME_COUNT_OPTIONS.some(f => jobMap.has(`${res}_${f}`))),
    [jobMap],
  );

  if (jobMap.size === 0) {
    return <EmptyState>No processing jobs for this folder yet. Configure and queue above.</EmptyState>;
  }

  return (
    <div className="border-border overflow-x-auto rounded-lg border p-2">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="text-muted-foreground px-2 py-1.5 text-left font-medium">Res \ Frames</th>
            {FRAME_COUNT_OPTIONS.map(f => (
              <th key={f} className="text-muted-foreground px-2 py-1.5 text-center font-medium">
                {f === 1 ? 'img' : `${f}f`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {activeResolutions.map(res => (
            <tr key={res} className="border-t border-white/10">
              <td className="px-2 py-1.5 font-medium">{res}px</td>
              {FRAME_COUNT_OPTIONS.map(f => {
                const job = jobMap.get(`${res}_${f}`);
                if (!job) {
                  return (
                    <td key={f} className="text-muted-foreground/30 px-2 py-1.5 text-center">
                      —
                    </td>
                  );
                }
                return (
                  <td key={f} className="px-2 py-1.5 text-center">
                    <StatusBadge status={job.status} progress={job.progress} />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
