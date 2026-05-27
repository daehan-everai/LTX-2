'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { ProcessingJob } from '@/lib/types';
import { apiFetch, apiPost } from '@/lib/api';

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

export function useJobs() {
  const queryClient = useQueryClient();

  const query = useQuery<ProcessingJob[]>({
    queryKey: ['jobs'],
    queryFn: () => apiFetch<ProcessingJob[]>('/api/jobs'),
    refetchInterval: query => {
      const data = query.state.data;
      if (!data || data.length === 0) return false;
      const allTerminal = data.every(j => TERMINAL_STATUSES.has(j.status));
      return allTerminal ? false : 1500;
    },
    refetchIntervalInBackground: false,
  });

  const createJob = useMutation({
    mutationFn: (job: { type: ProcessingJob['type']; name: string; config: Record<string, unknown> }) =>
      apiPost<ProcessingJob>('/api/jobs', job),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const stopJob = useMutation({
    mutationFn: (id: number) => apiPost<{ ok: boolean }>(`/api/jobs/${id}/stop`, {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  return {
    jobs: query.data || [],
    isError: query.isError,
    error: query.error,
    createJob: createJob.mutateAsync,
    stopJob: stopJob.mutateAsync,
    refreshJobs: query.refetch,
  };
}
