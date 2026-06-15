import type { TrainingConfig } from './types';

export function buildDefaultConfig(modelPath: string, textEncoderPath: string, outputDir: string): TrainingConfig {
  return {
    model: {
      modelPath,
      textEncoderPath,
      trainingMode: 'lora',
      loadCheckpoint: null,
    },
    lora: {
      rank: 48,
      alpha: 48,
      dropout: 0.05,
      targetModules: ['to_k', 'to_q', 'to_v', 'to_out.0', 'to_gate_logits', 'net.0.proj', 'net.2'],
    },
    trainingStrategy: {
      name: 'text_to_video',
      firstFrameConditioningP: 1.0,
      withAudio: true,
      audioLatentsDir: 'audio_latents',
      hFlip: true,
      firstFrameConditioningNoise: 0.0,
      temporalBoundaryLossWeight: 1.2,
      temporalBoundaryFrames: 3,
    },
    optimization: {
      learningRate: 8e-5,
      steps: 11200,
      batchSize: 1,
      gradientAccumulationSteps: 1,
      maxGradNorm: 1.0,
      optimizerType: 'muon',
      weightDecay: 0.0001,
      schedulerType: 'lambda_warmup',
      numWarmupSteps: 560,
      enableGradientCheckpointing: true,
      weightNoise: { mode: 'none', sigma: 0.01 },
    },
    flowMatching: {
      timestepSamplingMode: 'uniform',
      timestepLossWeighting: 'weighted',
    },
    validation: {
      prompts: [],
      images: [],
      negativePrompt: 'worst quality, inconsistent motion, blurry, jittery, distorted',
      videoDims: [416, 608, 241],
      frameRate: 24.0,
      seed: 42,
      inferenceSteps: 30,
      interval: 560,
      videosPerPrompt: 1,
      guidanceScale: 3.0,
      generateAudio: true,
      skipInitialValidation: false,
    },
    checkpoints: {
      interval: 560,
      keepLastN: -1,
      precision: 'bfloat16',
      saveTrainingState: 'full',
      resume: false,
    },
    outputDir: outputDir || '/tmp/ltx-training',
    seed: 42,
  };
}

export function deepMerge<T extends Record<string, unknown>>(defaults: T, partial: Record<string, unknown>): T {
  const result = { ...defaults };
  for (const key of Object.keys(partial)) {
    const val = partial[key];
    const def = (defaults as Record<string, unknown>)[key];
    if (
      val != null &&
      typeof val === 'object' &&
      !Array.isArray(val) &&
      def != null &&
      typeof def === 'object' &&
      !Array.isArray(def)
    ) {
      (result as Record<string, unknown>)[key] = deepMerge(
        def as Record<string, unknown>,
        val as Record<string, unknown>,
      );
    } else if (val !== undefined) {
      (result as Record<string, unknown>)[key] = val;
    }
  }
  return result;
}

export function extractTrainingConfig(
  jobConfig: Record<string, unknown>,
  defaults: TrainingConfig,
): {
  config: TrainingConfig;
  gpuMode: 'single' | 'ddp';
  gpuIds: string;
  datasetName: string;
} {
  const { gpuMode, gpuIds, datasetName, ...rest } = jobConfig;
  delete rest.configPath;
  delete rest.preprocessedDataRoot;

  return {
    config: deepMerge(defaults as unknown as Record<string, unknown>, rest) as unknown as TrainingConfig,
    gpuMode: (gpuMode as 'single' | 'ddp') || 'single',
    gpuIds: (gpuIds as string) || '0',
    datasetName: (datasetName as string) || '',
  };
}
