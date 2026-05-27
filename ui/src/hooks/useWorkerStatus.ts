'use client';

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api';

interface WorkerStatus {
  alive: boolean;
  lastSeen: string | null;
}

export function useWorkerStatus() {
  const query = useQuery<WorkerStatus>({
    queryKey: ['worker-status'],
    queryFn: () => apiFetch<WorkerStatus>('/api/worker/status'),
    refetchInterval: 3000,
    refetchIntervalInBackground: false,
  });

  return {
    alive: query.data?.alive ?? false,
    lastSeen: query.data?.lastSeen ?? null,
  };
}
