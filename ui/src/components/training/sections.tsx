'use client';

import type { TrainingConfig } from '@/lib/types';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Section, NumberField, TextField, SelectField, SwitchField, ListInput, VideoDimsField } from './fields';

type UpdateFn = (section: keyof TrainingConfig, patch: Record<string, unknown>) => void;

interface SectionProps {
  config: TrainingConfig;
  update: UpdateFn;
}

interface GeneralSectionProps {
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
}

// Video-only modules: self-attention (attn1), text cross-attention (attn2) and feed-forward (ff).
const VIDEO_CORE_MODULES = [
  'attn1.to_k',
  'attn1.to_q',
  'attn1.to_v',
  'attn1.to_out.0',
  'attn1.to_gate_logits',
  'attn2.to_k',
  'attn2.to_q',
  'attn2.to_v',
  'attn2.to_out.0',
  'attn2.to_gate_logits',
  'ff.net.0.proj',
  'ff.net.2',
];

// Audio-only modules: self-attention (audio_attn1), text cross-attention (audio_attn2) and feed-forward (audio_ff).
const AUDIO_CORE_MODULES = [
  'audio_attn1.to_k',
  'audio_attn1.to_q',
  'audio_attn1.to_v',
  'audio_attn1.to_out.0',
  'audio_attn1.to_gate_logits',
  'audio_attn2.to_k',
  'audio_attn2.to_q',
  'audio_attn2.to_v',
  'audio_attn2.to_out.0',
  'audio_attn2.to_gate_logits',
  'audio_ff.net.0.proj',
  'audio_ff.net.2',
];

// Cross-modal bridge: Q from video, K/V from audio — lets video attend to audio (audio signal impacts video).
const AUDIO_TO_VIDEO_BRIDGE_MODULES = [
  'audio_to_video_attn.to_k',
  'audio_to_video_attn.to_q',
  'audio_to_video_attn.to_v',
  'audio_to_video_attn.to_out.0',
  'audio_to_video_attn.to_gate_logits',
];

// Cross-modal bridge: Q from audio, K/V from video — lets audio attend to video (video signal impacts audio).
const VIDEO_TO_AUDIO_BRIDGE_MODULES = [
  'video_to_audio_attn.to_k',
  'video_to_audio_attn.to_q',
  'video_to_audio_attn.to_v',
  'video_to_audio_attn.to_out.0',
  'video_to_audio_attn.to_gate_logits',
];

const TARGET_MODULE_PRESETS = [
  {
    id: 'full',
    label: 'Full',
    description: 'All modules — broad patterns match video, audio & cross-modal',
    modules: ['to_k', 'to_q', 'to_v', 'to_out.0', 'to_gate_logits', 'net.0.proj', 'net.2'],
  },
  {
    id: 'video',
    label: 'Video',
    description: 'Video-only: self-attention, text cross-attention & feed-forward — no audio or cross-modal modules',
    modules: VIDEO_CORE_MODULES,
  },
  {
    id: 'audio',
    label: 'Audio',
    description: 'Audio-only: self-attention, text cross-attention & feed-forward — no video or cross-modal modules',
    modules: AUDIO_CORE_MODULES,
  },
  {
    id: 'video_bridge',
    label: 'Video + Bridge',
    description:
      'Video modules + the audio→video cross-attention bridge (Q from video, K/V from audio) so audio can attend ' +
      'into video — i.e. the audio signal impacts the video. No audio-core modules.',
    modules: [...VIDEO_CORE_MODULES, ...AUDIO_TO_VIDEO_BRIDGE_MODULES],
  },
  {
    id: 'audio_bridge',
    label: 'Audio + Bridge',
    description:
      'Audio modules + the video→audio cross-attention bridge (Q from audio, K/V from video) so video can attend ' +
      'into audio — i.e. the video signal impacts the audio. No video-core modules.',
    modules: [...AUDIO_CORE_MODULES, ...VIDEO_TO_AUDIO_BRIDGE_MODULES],
  },
];

export function ModelSection({ config, update }: SectionProps) {
  return (
    <Section title="Model">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <TextField
            label="Checkpoint Path"
            value={config.model.modelPath}
            onChange={v => update('model', { modelPath: v })}
            mono
          />
        </div>
        <div className="sm:col-span-2">
          <TextField
            label="Text Encoder Path"
            value={config.model.textEncoderPath}
            onChange={v => update('model', { textEncoderPath: v })}
            mono
          />
        </div>
        <div className="sm:col-span-2">
          <TextField
            label="Load Checkpoint (optional)"
            value={config.model.loadCheckpoint || ''}
            onChange={v => update('model', { loadCheckpoint: v || null })}
            placeholder="None — resume from a previous checkpoint path"
            mono
          />
        </div>
      </div>
    </Section>
  );
}

export function LoraSection({ config, update }: SectionProps) {
  const modules = config.lora.targetModules;
  const sortedCurrent = [...modules].sort().join(',');
  const activePreset = TARGET_MODULE_PRESETS.find(p => [...p.modules].sort().join(',') === sortedCurrent);

  return (
    <Section title="LoRA">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <NumberField label="Rank" value={config.lora.rank} onChange={v => update('lora', { rank: v })} />
        <NumberField label="Alpha" value={config.lora.alpha} onChange={v => update('lora', { alpha: v })} />
        <NumberField
          label="Dropout"
          value={config.lora.dropout}
          onChange={v => update('lora', { dropout: v })}
          step={0.01}
        />
      </div>
      <SwitchField
        label="Freeze Extra Modules"
        checked={config.lora.freezeExtraModules}
        onChange={v => update('lora', { freezeExtraModules: v })}
      />
      <div className="space-y-2">
        <Label className="text-xs">Target Layers</Label>
        <div className="flex flex-wrap gap-2">
          {TARGET_MODULE_PRESETS.map(preset => (
            <Button
              key={preset.id}
              type="button"
              size="sm"
              variant={activePreset?.id === preset.id ? 'default' : 'outline'}
              onClick={() => update('lora', { targetModules: [...preset.modules] })}
              title={preset.description}
              className="h-7 px-3 text-xs"
            >
              {preset.label}
            </Button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {modules.map(mod => (
            <Badge key={mod} variant="secondary" className="font-mono text-[10px]">
              {mod}
            </Badge>
          ))}
        </div>
      </div>
    </Section>
  );
}

export function StrategySection({ config, update }: SectionProps) {
  return (
    <Section title="Training Strategy">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <NumberField
          label="First Frame Cond. P"
          value={config.trainingStrategy.firstFrameConditioningP}
          onChange={v => update('trainingStrategy', { firstFrameConditioningP: v })}
          step={0.1}
        />
        <NumberField
          label="Boundary Loss Weight"
          value={config.trainingStrategy.temporalBoundaryLossWeight}
          onChange={v => update('trainingStrategy', { temporalBoundaryLossWeight: v })}
          step={0.1}
        />
        <NumberField
          label="Boundary Frames"
          value={config.trainingStrategy.temporalBoundaryFrames}
          onChange={v => update('trainingStrategy', { temporalBoundaryFrames: v })}
        />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        <SwitchField
          label="Audio"
          checked={config.trainingStrategy.withAudio}
          onChange={v => update('trainingStrategy', { withAudio: v })}
        />
        <SwitchField
          label="H-Flip Augmentation"
          checked={config.trainingStrategy.hFlip}
          onChange={v => update('trainingStrategy', { hFlip: v })}
        />
      </div>
    </Section>
  );
}

export function OptimizationSection({ config, update }: SectionProps) {
  return (
    <Section title="Optimization">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <NumberField
          label="Learning Rate"
          value={config.optimization.learningRate}
          onChange={v => update('optimization', { learningRate: v })}
          step={0.000001}
          mono
        />
        <NumberField
          label="Steps"
          value={config.optimization.steps}
          onChange={v => update('optimization', { steps: v })}
        />
        <NumberField
          label="Batch Size"
          value={config.optimization.batchSize}
          onChange={v => update('optimization', { batchSize: v })}
        />
        <NumberField
          label="Grad Accumulation"
          value={config.optimization.gradientAccumulationSteps}
          onChange={v => update('optimization', { gradientAccumulationSteps: v })}
        />
        <NumberField
          label="Max Grad Norm"
          value={config.optimization.maxGradNorm}
          onChange={v => update('optimization', { maxGradNorm: v })}
          step={0.1}
        />
        <NumberField
          label="Warmup Steps"
          value={config.optimization.numWarmupSteps}
          onChange={v => update('optimization', { numWarmupSteps: v })}
        />
        <NumberField
          label="Weight Decay"
          value={config.optimization.weightDecay}
          onChange={v => update('optimization', { weightDecay: v })}
          step={0.0001}
          mono
        />
        <SelectField
          label="Optimizer"
          value={config.optimization.optimizerType}
          onChange={v => update('optimization', { optimizerType: v })}
          options={[
            { value: 'muon', label: 'Muon' },
            { value: 'adamw', label: 'AdamW' },
            { value: 'adamw8bit', label: 'AdamW 8-bit' },
          ]}
        />
        <SelectField
          label="Scheduler"
          value={config.optimization.schedulerType}
          onChange={v => update('optimization', { schedulerType: v })}
          options={[
            { value: 'cosine', label: 'Cosine' },
            { value: 'constant', label: 'Constant' },
            { value: 'lambda_warmup', label: 'Lambda Warmup' },
          ]}
        />
      </div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <SelectField
          label="Weight Noise"
          value={config.optimization.weightNoise.mode}
          onChange={v => update('optimization', { weightNoise: { ...config.optimization.weightNoise, mode: v } })}
          options={[
            { value: 'none', label: 'None' },
            { value: 'relative', label: 'Relative' },
            { value: 'absolute', label: 'Absolute' },
          ]}
        />
        {config.optimization.weightNoise.mode !== 'none' && (
          <NumberField
            label="Sigma"
            value={config.optimization.weightNoise.sigma}
            onChange={v => update('optimization', { weightNoise: { ...config.optimization.weightNoise, sigma: v } })}
            step={0.001}
            mono
          />
        )}
        <SwitchField
          label="Gradient Checkpointing"
          checked={config.optimization.enableGradientCheckpointing}
          onChange={v => update('optimization', { enableGradientCheckpointing: v })}
        />
      </div>
    </Section>
  );
}

export function ValidationSection({ config, update }: SectionProps) {
  return (
    <Section title="Validation" defaultOpen={false}>
      <div className="space-y-4">
        <div className="space-y-2">
          <Label className="text-xs">Prompts (one per line)</Label>
          <ListInput
            value={config.validation.prompts}
            onChange={v => update('validation', { prompts: v })}
            placeholder="A serene mountain lake at sunrise..."
          />
        </div>
        <div className="space-y-2">
          <Label className="text-xs">Conditioning Images (one path per line)</Label>
          <ListInput
            value={config.validation.images}
            onChange={v => update('validation', { images: v })}
            placeholder="/path/to/image.jpeg"
          />
        </div>
        <TextField
          label="Negative Prompt"
          value={config.validation.negativePrompt}
          onChange={v => update('validation', { negativePrompt: v })}
        />
        <VideoDimsField value={config.validation.videoDims} onChange={v => update('validation', { videoDims: v })} />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <NumberField
            label="Inference Steps"
            value={config.validation.inferenceSteps}
            onChange={v => update('validation', { inferenceSteps: v })}
          />
          <NumberField
            label="Interval"
            value={config.validation.interval}
            onChange={v => update('validation', { interval: v })}
          />
          <NumberField
            label="Guidance Scale"
            value={config.validation.guidanceScale}
            onChange={v => update('validation', { guidanceScale: v })}
            step={0.5}
          />
          <NumberField label="Seed" value={config.validation.seed} onChange={v => update('validation', { seed: v })} />
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <SwitchField
            label="Generate Audio"
            checked={config.validation.generateAudio}
            onChange={v => update('validation', { generateAudio: v })}
          />
          <SwitchField
            label="Skip Initial Validation"
            checked={config.validation.skipInitialValidation}
            onChange={v => update('validation', { skipInitialValidation: v })}
          />
        </div>
      </div>
    </Section>
  );
}

export function CheckpointsSection({ config, update }: SectionProps) {
  return (
    <Section title="Checkpoints" defaultOpen={false}>
      <div className="grid grid-cols-2 gap-4">
        <NumberField
          label="Save Interval"
          value={config.checkpoints.interval}
          onChange={v => update('checkpoints', { interval: v })}
        />
        <NumberField
          label="Keep Last N (-1 = all)"
          value={config.checkpoints.keepLastN}
          onChange={v => update('checkpoints', { keepLastN: v })}
        />
        <SelectField
          label="Precision"
          value={config.checkpoints.precision}
          onChange={v => update('checkpoints', { precision: v })}
          options={[
            { value: 'bfloat16', label: 'bfloat16' },
            { value: 'float16', label: 'float16' },
            { value: 'float32', label: 'float32' },
          ]}
        />
        <SelectField
          label="Save Training State"
          value={config.checkpoints.saveTrainingState}
          onChange={v => update('checkpoints', { saveTrainingState: v })}
          options={[
            { value: 'full', label: 'Full (optimizer + scheduler + RNG)' },
            { value: 'minimal', label: 'Minimal (scheduler + RNG only)' },
            { value: 'off', label: 'Off' },
          ]}
        />
      </div>
      <SwitchField
        label="Resume training state from the loaded checkpoint"
        checked={config.checkpoints.resume}
        onChange={v => update('checkpoints', { resume: v })}
      />
    </Section>
  );
}

export function FlowMatchingSection({ config, update }: SectionProps) {
  return (
    <Section title="Flow Matching">
      <div className="grid grid-cols-2 gap-4">
        <SelectField
          label="Timestep Sampling"
          value={config.flowMatching.timestepSamplingMode}
          onChange={v => update('flowMatching', { timestepSamplingMode: v })}
          options={[
            { value: 'uniform', label: 'Uniform' },
            { value: 'shifted_logit_normal', label: 'Shifted Logit Normal' },
          ]}
        />
        <SelectField
          label="Loss Weighting"
          value={config.flowMatching.timestepLossWeighting}
          onChange={v => update('flowMatching', { timestepLossWeighting: v })}
          options={[
            { value: 'none', label: 'None' },
            { value: 'bell', label: 'Bell' },
            { value: 'weighted', label: 'Weighted' },
          ]}
        />
      </div>
    </Section>
  );
}

export function GeneralSection({ config, onChange }: GeneralSectionProps) {
  return (
    <Section title="General" defaultOpen={false}>
      <div className="grid grid-cols-2 gap-4">
        <TextField
          label="Output Directory"
          value={config.outputDir}
          onChange={v => onChange({ ...config, outputDir: v })}
          mono
        />
        <NumberField label="Seed" value={config.seed} onChange={v => onChange({ ...config, seed: v })} />
      </div>
    </Section>
  );
}
