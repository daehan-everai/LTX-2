'use client';

import { useState, useEffect, useRef, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import type { TrainingConfig, ProcessingJob } from '@/lib/types';
import { buildDefaultConfig, extractTrainingConfig } from '@/lib/training';
import { useDatasets } from '@/hooks/useDatasets';
import { useRuns } from '@/hooks/useRuns';
import { useSettings } from '@/hooks/useSettings';
import { apiFetch } from '@/lib/api';
import { TrainingConfigForm } from '@/components/TrainingConfigForm';
import { PageHeader } from '@/components/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Play, Loader2 } from 'lucide-react';

export default function TrainingPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center p-12">
          <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
        </div>
      }
    >
      <TrainingPageInner />
    </Suspense>
  );
}

function TrainingPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fromJobId = searchParams.get('fromJob');

  const { settings } = useSettings();
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [gpuMode, setGpuMode] = useState<'single' | 'ddp'>('single');
  const [gpuIds, setGpuIds] = useState('0');
  const [startError, setStartError] = useState<string | null>(null);

  const { datasets } = useDatasets();
  const { createRun, isCreating } = useRuns();
  const appliedFromJob = useRef<string | null>(null);

  useEffect(() => {
    if (fromJobId) return;
    if (settings && !config) {
      setConfig(
        buildDefaultConfig(
          settings.modelPath || '',
          settings.textEncoderPath || '',
          settings.outputDir || '/tmp/ltx-training',
        ),
      );
    }
  }, [fromJobId, settings, config]);

  useEffect(() => {
    if (!fromJobId || !settings) return;
    if (appliedFromJob.current === fromJobId) return;
    appliedFromJob.current = fromJobId;

    apiFetch<ProcessingJob>(`/api/jobs/${fromJobId}`)
      .then(job => {
        if (job.type !== 'training') return;
        const defaults = buildDefaultConfig(
          settings.modelPath || '',
          settings.textEncoderPath || '',
          settings.outputDir || '/tmp/ltx-training',
        );
        const restored = extractTrainingConfig(job.config as Record<string, unknown>, defaults);
        setConfig(restored.config);
        setGpuMode(restored.gpuMode);
        setGpuIds(restored.gpuIds);
        if (restored.datasetName) setSelectedDataset(restored.datasetName);
      })
      .catch(() => {
        setConfig(
          buildDefaultConfig(
            settings.modelPath || '',
            settings.textEncoderPath || '',
            settings.outputDir || '/tmp/ltx-training',
          ),
        );
      });
  }, [fromJobId, settings]);

  if (!config) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
      </div>
    );
  }

  const activeDataset = selectedDataset || datasets[0]?.name || '';

  const handleStartTraining = async () => {
    setStartError(null);
    const outputName = config.outputDir.replace(/\/$/, '').split('/').pop() || 'training-run';
    try {
      await createRun({
        name: outputName,
        config: config as unknown as Record<string, unknown>,
        gpuMode,
        gpuIds,
        datasetName: activeDataset,
      });
      router.push('/runs');
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="space-y-6 p-3 md:p-4">
      <PageHeader title="Training" subtitle="Configure and launch a training run" />

      <div className="grid gap-6 xl:grid-cols-[1fr_300px]">
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-lg">Run Setup</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label className="text-xs">Training Dataset</Label>
                  <Select value={activeDataset} onValueChange={v => v && setSelectedDataset(v)}>
                    <SelectTrigger className="w-full text-xs">
                      <SelectValue placeholder="Select a dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.map(ds => (
                        <SelectItem key={ds.name} value={ds.name}>
                          {ds.name} ({ds.buckets.length} buckets)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">GPU Configuration</Label>
                  <div className="flex gap-2">
                    <Select
                      value={gpuMode}
                      onValueChange={v => {
                        if (!v) return;
                        const mode = v as 'single' | 'ddp';
                        setGpuMode(mode);
                        setGpuIds(mode === 'single' ? '0' : '0,1');
                      }}
                    >
                      <SelectTrigger className="w-[140px] text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="single">Single GPU</SelectItem>
                        <SelectItem value="ddp">Multi-GPU (DDP)</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={gpuIds} onValueChange={v => v && setGpuIds(v)}>
                      <SelectTrigger className="flex-1 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {gpuMode === 'single' ? (
                          <>
                            <SelectItem value="0">GPU 0</SelectItem>
                            <SelectItem value="1">GPU 1</SelectItem>
                            <SelectItem value="2">GPU 2</SelectItem>
                            <SelectItem value="3">GPU 3</SelectItem>
                          </>
                        ) : (
                          <>
                            <SelectItem value="0,1">GPU 0, 1</SelectItem>
                            <SelectItem value="0,1,2">GPU 0, 1, 2</SelectItem>
                            <SelectItem value="0,1,2,3">GPU 0, 1, 2, 3</SelectItem>
                          </>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <TrainingConfigForm config={config} onChange={setConfig} />
        </div>

        <div className="space-y-4">
          <Card className="sticky top-6">
            <CardHeader className="py-3">
              <CardTitle className="text-lg">Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Dataset</span>
                <Badge variant="outline" className="text-[10px]">
                  {activeDataset || '—'}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Mode</span>
                <Badge variant="secondary" className="text-[10px]">
                  {config.model.trainingMode === 'full' ? 'Full' : 'LoRA'}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">LoRA Rank</span>
                <span>{config.lora.rank}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Steps</span>
                <span>{config.optimization.steps.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">LR</span>
                <span className="font-mono">{config.optimization.learningRate}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Optimizer</span>
                <span>{config.optimization.optimizerType}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">GPU</span>
                <span>{gpuMode === 'ddp' ? `DDP (${gpuIds})` : `GPU ${gpuIds}`}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Audio</span>
                <Badge variant={config.trainingStrategy.withAudio ? 'default' : 'secondary'} className="text-[10px]">
                  {config.trainingStrategy.withAudio ? 'Yes' : 'No'}
                </Badge>
              </div>

              {startError && <p className="text-destructive text-xs">{startError}</p>}

              <Button className="w-full" onClick={handleStartTraining} disabled={isCreating || !activeDataset}>
                <Play className="mr-1.5 h-4 w-4" />
                {isCreating ? 'Starting...' : 'Start Training'}
              </Button>

              {!activeDataset && (
                <p className="text-muted-foreground text-center text-[10px]">Select a dataset to enable training</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
