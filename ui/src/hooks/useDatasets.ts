'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { TrainingDataset, DatasetBucket } from '@/lib/types';
import { apiFetch, apiPost, apiDelete } from '@/lib/api';

export function useDatasets() {
  const queryClient = useQueryClient();

  const query = useQuery<TrainingDataset[]>({
    queryKey: ['datasets'],
    queryFn: () => apiFetch<TrainingDataset[]>('/api/datasets'),
    refetchInterval: query => {
      const data = query.state.data;
      const hasActiveBuilds = data?.some(d => d.buildStatus === 'queued' || d.buildStatus === 'running');
      return hasActiveBuilds ? 3000 : false;
    },
    refetchIntervalInBackground: false,
  });

  const createDataset = useMutation({
    mutationFn: (dataset: { name: string; path: string; buckets: DatasetBucket[] }) =>
      apiPost<TrainingDataset>('/api/datasets', dataset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const deleteDataset = useMutation({
    mutationFn: (id: number) => apiDelete<{ ok: boolean }>(`/api/datasets?id=${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
  });

  return {
    datasets: query.data || [],
    isError: query.isError,
    error: query.error,
    createDataset: createDataset.mutateAsync,
    deleteDataset: deleteDataset.mutateAsync,
    refreshDatasets: query.refetch,
  };
}
