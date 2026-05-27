'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { AppSettings } from '@/lib/types';
import { apiFetch, apiPut } from '@/lib/api';

export function useSettings() {
  const queryClient = useQueryClient();

  const query = useQuery<AppSettings>({
    queryKey: ['settings'],
    queryFn: () => apiFetch<AppSettings>('/api/settings'),
  });

  const mutation = useMutation({
    mutationFn: (settings: Partial<AppSettings>) => apiPut<{ ok: boolean }>('/api/settings', settings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });

  return {
    settings: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    saveSettings: mutation.mutateAsync,
    isSaving: mutation.isPending,
    saveError: mutation.error as Error | null,
  };
}
