'use client';

import type { TrainingConfig } from '@/lib/types';
import {
  ModelSection,
  LoraSection,
  StrategySection,
  OptimizationSection,
  FlowMatchingSection,
  ValidationSection,
  CheckpointsSection,
  GeneralSection,
} from './training/sections';

interface Props {
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
}

export function TrainingConfigForm({ config, onChange }: Props) {
  const update = <K extends keyof TrainingConfig>(
    section: K,
    patch: Partial<TrainingConfig[K] & Record<string, unknown>>,
  ) => {
    const current = config[section];
    if (typeof current === 'object' && current !== null) {
      onChange({
        ...config,
        [section]: { ...(current as object), ...patch },
      });
    }
  };

  return (
    <div className="space-y-3">
      <ModelSection config={config} update={update} />
      <LoraSection config={config} update={update} />
      <StrategySection config={config} update={update} />
      <OptimizationSection config={config} update={update} />
      <FlowMatchingSection config={config} update={update} />
      <ValidationSection config={config} update={update} />
      <CheckpointsSection config={config} update={update} />
      <GeneralSection config={config} onChange={onChange} />
    </div>
  );
}
