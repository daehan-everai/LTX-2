'use client';

import { useEffect, useMemo, useState } from 'react';
import type { SourceFolder, ResolutionOption, FrameCountOption } from '@/lib/types';
import { RESOLUTION_OPTIONS, FRAME_COUNT_OPTIONS } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { Plus } from 'lucide-react';

export interface ResFrameConfig {
  hFlip: boolean;
  frameSampling: 'uniform' | 'head';
  withAudio: boolean;
  datasetFilename: string;
  /** Dataset column with reference video paths (IC-LoRA). Empty = disabled. */
  referenceColumn: string;
  referenceDownscaleFactor: number;
}

interface Props {
  folder: SourceFolder;
  onQueueProcessing: (
    configs: Array<{
      resolution: number;
      frameCount: number;
      config: ResFrameConfig;
    }>,
  ) => void;
}

export function FolderConfigPanel({ folder, onQueueProcessing }: Props) {
  const [selectedResolutions, setSelectedResolutions] = useState<Set<ResolutionOption>>(new Set());
  const [selectedFrames, setSelectedFrames] = useState<Record<number, Set<FrameCountOption>>>({});
  const [configs, setConfigs] = useState<Record<string, ResFrameConfig>>({});

  // Reset selections when switching to a different folder so jobs are not
  // queued with another folder's resolution/frame configuration.
  useEffect(() => {
    setSelectedResolutions(new Set());
    setSelectedFrames({});
    setConfigs({});
  }, [folder.id]);

  const toggleResolution = (res: ResolutionOption) => {
    setSelectedResolutions(prev => {
      const next = new Set(prev);
      if (next.has(res)) next.delete(res);
      else next.add(res);
      return next;
    });
  };

  const toggleFrame = (res: number, frame: FrameCountOption) => {
    setSelectedFrames(prev => {
      const resFrames = new Set(prev[res] || []);
      if (resFrames.has(frame)) resFrames.delete(frame);
      else resFrames.add(frame);
      return { ...prev, [res]: resFrames };
    });

    const key = `${res}_${frame}`;
    if (!configs[key]) {
      const highFrameCount = frame >= 100;
      setConfigs(prev => ({
        ...prev,
        [key]: {
          hFlip: true,
          frameSampling: highFrameCount ? 'uniform' : 'head',
          withAudio: highFrameCount,
          datasetFilename: 'dataset.json',
          referenceColumn: '',
          referenceDownscaleFactor: 1,
        },
      }));
    }
  };

  const updateConfig = (key: string, patch: Partial<ResFrameConfig>) => {
    setConfigs(prev => ({
      ...prev,
      [key]: { ...prev[key], ...patch },
    }));
  };

  const activeResolutions = useMemo(() => Array.from(selectedResolutions).sort((a, b) => a - b), [selectedResolutions]);

  const allPairs = useMemo(() => {
    const pairs: Array<{ resolution: number; frameCount: number; config: ResFrameConfig }> = [];
    for (const res of activeResolutions) {
      const frames = selectedFrames[res] || new Set();
      for (const f of Array.from(frames).sort((a, b) => a - b)) {
        const key = `${res}_${f}`;
        if (configs[key]) {
          pairs.push({ resolution: res, frameCount: f, config: configs[key] });
        }
      }
    }
    return pairs;
  }, [activeResolutions, selectedFrames, configs]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">
          Configure Processing: <span className="font-mono text-base break-all">{folder.path}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="space-y-2">
          <Label className="text-muted-foreground text-xs tracking-wider uppercase">Target Resolutions</Label>
          <div className="flex flex-wrap gap-2">
            {RESOLUTION_OPTIONS.map(res => (
              <button
                type="button"
                key={res}
                onClick={() => toggleResolution(res)}
                className={cn(
                  'rounded-md border px-3 py-1.5 text-sm font-semibold transition-colors',
                  selectedResolutions.has(res)
                    ? 'border-primary bg-primary/20 text-primary'
                    : 'border-border bg-background text-muted-foreground hover:border-primary/40',
                )}
              >
                {res}
              </button>
            ))}
          </div>
        </div>

        {activeResolutions.map(res => {
          const resFrames = selectedFrames[res] || new Set<FrameCountOption>();

          return (
            <div key={res} className="border-border space-y-2 rounded-lg border p-3">
              <div className="flex items-center gap-3">
                <Label className="shrink-0 text-sm font-medium">{res}px</Label>
                <div className="flex flex-wrap gap-1.5">
                  {FRAME_COUNT_OPTIONS.map(frame => (
                    <button
                      type="button"
                      key={frame}
                      onClick={() => toggleFrame(res, frame)}
                      className={cn(
                        'rounded-md border px-2 py-0.5 text-xs font-medium transition-colors',
                        resFrames.has(frame)
                          ? 'border-primary bg-primary/20 text-primary'
                          : 'border-border bg-background text-muted-foreground hover:border-primary/40',
                      )}
                    >
                      {frame === 1 ? '1 (image)' : `${frame}f`}
                    </button>
                  ))}
                </div>
              </div>

              {Array.from(resFrames)
                .sort((a, b) => a - b)
                .map(frame => {
                  const key = `${res}_${frame}`;
                  const cfg = configs[key];
                  if (!cfg) return null;

                  return (
                    <div
                      key={key}
                      className="bg-muted/50 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md px-3 py-2"
                    >
                      <Badge variant="outline" className="w-[100px] shrink-0 justify-center text-xs">
                        {res} x {frame === 1 ? 'img' : `${frame}f`}
                      </Badge>

                      <Input
                        className="h-8 min-w-[140px] flex-1 font-mono text-xs"
                        value={cfg.datasetFilename}
                        onChange={e => updateConfig(key, { datasetFilename: e.target.value })}
                      />

                      <div className="flex items-center gap-1.5">
                        <Switch
                          id={`${key}-reference`}
                          checked={cfg.referenceColumn.trim() !== ''}
                          onCheckedChange={v => updateConfig(key, { referenceColumn: v ? 'reference_path' : '' })}
                        />
                        <Label htmlFor={`${key}-reference`} className="text-xs">
                          Reference
                        </Label>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <Switch
                          id={`${key}-hflip`}
                          checked={cfg.hFlip}
                          onCheckedChange={v => updateConfig(key, { hFlip: v })}
                        />
                        <Label htmlFor={`${key}-hflip`} className="text-xs">
                          H-Flip
                        </Label>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <Switch
                          id={`${key}-audio`}
                          checked={cfg.withAudio}
                          onCheckedChange={v => updateConfig(key, { withAudio: v })}
                        />
                        <Label htmlFor={`${key}-audio`} className="text-xs">
                          Audio
                        </Label>
                      </div>

                      <Select
                        value={cfg.frameSampling}
                        onValueChange={v => updateConfig(key, { frameSampling: v as 'uniform' | 'head' })}
                      >
                        <SelectTrigger className="h-8 w-[100px] text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="uniform">Uniform</SelectItem>
                          <SelectItem value="head">Head</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  );
                })}
            </div>
          );
        })}

        {allPairs.length > 0 && (
          <div className="flex items-center justify-between pt-2">
            <span className="text-muted-foreground text-sm">
              {allPairs.length} processing job{allPairs.length !== 1 ? 's' : ''} to queue
            </span>
            <Button onClick={() => onQueueProcessing(allPairs)}>
              <Plus className="mr-1.5 h-4 w-4" />
              Queue Processing
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
