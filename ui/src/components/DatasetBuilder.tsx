'use client';

import { useState } from 'react';
import type { ProcessingJob, TrainingDataset, DatasetBucket } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { EmptyState } from '@/components/EmptyState';
import { cn } from '@/lib/utils';
import {
  Package,
  Plus,
  CheckCircle2,
  Square,
  CheckSquare,
  Trash2,
  AlertTriangle,
  RefreshCw,
  Loader2,
  Music,
} from 'lucide-react';

interface Props {
  completedJobs: ProcessingJob[];
  datasets: TrainingDataset[];
  onBuildDataset: (name: string, buckets: DatasetBucket[]) => void | Promise<void>;
  onDeleteDataset: (id: number) => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
}

export function DatasetBuilder({
  completedJobs,
  datasets,
  onBuildDataset,
  onDeleteDataset,
  onRefresh,
  isRefreshing,
}: Props) {
  const [datasetName, setDatasetName] = useState('');
  const [selectedJobIds, setSelectedJobIds] = useState<Set<number>>(new Set());

  const toggleJob = (id: number) => {
    setSelectedJobIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleBuild = async () => {
    if (!datasetName.trim()) return;
    const buckets: DatasetBucket[] = completedJobs
      .filter(j => selectedJobIds.has(j.id))
      .map(j => {
        const cfg = j.config as {
          folderId?: number;
          outputFolderPath?: string;
          resolution?: number;
          frameCounts?: number[];
          withAudio?: boolean;
          hFlip?: boolean;
          resolutionBuckets?: string;
          audioOnly?: boolean;
        };
        const bucketKeys = cfg.resolutionBuckets ? cfg.resolutionBuckets.split(';') : [];
        const folderPath = cfg.outputFolderPath || '';
        return {
          folderName: folderPath.split('/').filter(Boolean).pop() || 'unknown',
          folderPath,
          jobId: j.id,
          resolution: cfg.resolution || 0,
          frameCount: (cfg.frameCounts || [0])[0],
          bucketKeys,
          hasAudio: cfg.audioOnly || cfg.withAudio || false,
          hasHFlip: cfg.hFlip || false,
          isAudioOnly: cfg.audioOnly || false,
        };
      });
    try {
      await onBuildDataset(datasetName, buckets);
      setDatasetName('');
      setSelectedJobIds(new Set());
    } catch {
      // Error is surfaced by the parent via buildError state
    }
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-muted-foreground text-xs tracking-wider uppercase">Existing Training Datasets</Label>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1.5 text-xs"
            onClick={onRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={cn('h-3 w-3', isRefreshing && 'animate-spin')} />
            Refresh
          </Button>
        </div>
        {datasets.length === 0 ? (
          <EmptyState className="p-4">No training datasets yet.</EmptyState>
        ) : (
          <>
            {datasets.map(ds => {
              const isBuilding = ds.buildStatus === 'queued' || ds.buildStatus === 'running';
              const isMissing = !isBuilding && !ds.pathExists;
              return (
                <Card
                  key={ds.id}
                  className={cn(
                    'bg-muted/30',
                    isMissing && 'border-destructive/30 bg-destructive/5',
                    isBuilding && 'border-blue-500/30 bg-blue-500/5',
                  )}
                >
                  <CardContent className="flex items-center gap-3 p-3">
                    {isBuilding ? (
                      <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-400" />
                    ) : (
                      <Package
                        className={cn('h-4 w-4 shrink-0', ds.pathExists ? 'text-emerald-400' : 'text-destructive/70')}
                      />
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{ds.name}</span>
                        {isBuilding && (
                          <span className="text-[10px] text-blue-400">
                            {ds.buildStatus === 'running' ? 'building…' : 'queued…'}
                          </span>
                        )}
                        {isMissing && (
                          <span className="text-destructive flex items-center gap-1 text-[10px]">
                            <AlertTriangle className="h-3 w-3" />
                            path missing
                          </span>
                        )}
                      </div>
                      <p className="text-muted-foreground font-mono text-xs break-all">{ds.path}</p>
                    </div>
                    <Badge variant="secondary" className="shrink-0 text-[10px]">
                      {ds.buckets.length} bucket{ds.buckets.length !== 1 ? 's' : ''}
                    </Badge>
                    <ConfirmDialog
                      title="Delete dataset?"
                      description={`This will remove the "${ds.name}" dataset record. The files on disk will not be deleted.`}
                      confirmLabel="Delete"
                      onConfirm={() => onDeleteDataset(ds.id)}
                    >
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-muted-foreground hover:bg-destructive/15 hover:text-destructive h-7 w-7 shrink-0 p-0"
                        title="Delete dataset record"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </ConfirmDialog>
                  </CardContent>
                </Card>
              );
            })}
          </>
        )}
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Build Training Dataset</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="dsName" className="text-xs">
              Dataset Name
            </Label>
            <Input
              id="dsName"
              placeholder="e.g. nature-multi-res"
              value={datasetName}
              onChange={e => setDatasetName(e.target.value)}
            />
          </div>

          {completedJobs.length === 0 ? (
            <EmptyState>No completed preprocessing jobs. Process some folders first.</EmptyState>
          ) : (
            <div className="space-y-2">
              <Label className="text-muted-foreground text-xs tracking-wider uppercase">Select Processed Buckets</Label>
              <div className="max-h-64 space-y-1.5 overflow-y-auto">
                {completedJobs.map(job => {
                  const checked = selectedJobIds.has(job.id);
                  const isAudioOnly = Boolean((job.config as { audioOnly?: boolean }).audioOnly);
                  return (
                    <button
                      key={job.id}
                      type="button"
                      onClick={() => toggleJob(job.id)}
                      className={cn(
                        'flex w-full cursor-pointer items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors',
                        checked ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/30',
                      )}
                    >
                      {checked ? (
                        <CheckSquare className="text-primary h-4 w-4 shrink-0" />
                      ) : (
                        <Square className="text-muted-foreground h-4 w-4 shrink-0" />
                      )}
                      {isAudioOnly ? (
                        <Music className="h-3.5 w-3.5 shrink-0 text-violet-400" />
                      ) : (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
                      )}
                      <span className="text-sm">{job.name}</span>
                      {isAudioOnly && (
                        <Badge variant="secondary" className="ml-auto shrink-0 text-[10px]">
                          Audio Only
                        </Badge>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {selectedJobIds.size > 0 && datasetName.trim() && (
            <div className="flex items-center justify-between pt-2">
              <span className="text-muted-foreground text-sm">
                {selectedJobIds.size} bucket{selectedJobIds.size !== 1 ? 's' : ''} selected
              </span>
              <Button onClick={handleBuild}>
                <Plus className="mr-1.5 h-4 w-4" />
                Build Dataset
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
