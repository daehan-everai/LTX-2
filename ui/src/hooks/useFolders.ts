'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { SourceFolder } from '@/lib/types';
import { apiFetch, apiPost, apiPatch, apiDelete } from '@/lib/api';

export function useFolders() {
  const queryClient = useQueryClient();

  const query = useQuery<SourceFolder[]>({
    queryKey: ['folders'],
    queryFn: () => apiFetch<SourceFolder[]>('/api/folders'),
  });

  const addFolder = useMutation({
    mutationFn: (folderPath: string) => apiPost<SourceFolder>('/api/folders', { path: folderPath }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
    },
  });

  const removeFolder = useMutation({
    mutationFn: (id: number) => apiDelete<{ ok: boolean }>(`/api/folders?id=${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
    },
  });

  const refreshFolderMutation = useMutation({
    mutationFn: (id: number) => apiPatch<SourceFolder>(`/api/folders?id=${id}`, {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  return {
    folders: query.data || [],
    isError: query.isError,
    error: query.error,
    addFolder: addFolder.mutateAsync,
    removeFolder: removeFolder.mutateAsync,
    refreshFolder: refreshFolderMutation.mutateAsync,
  };
}
