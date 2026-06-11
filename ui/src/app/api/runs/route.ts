import { NextResponse } from 'next/server';
import { db } from '@/db';
import { jobs, trainingDatasets } from '@/db/schema';
import { nextQueuePosition } from '@/db/queries';
import { eq } from 'drizzle-orm';
import { parseJobConfig, toErrorMessage } from '@/lib/utils';
import fs from 'fs';
import path from 'path';
import YAML from 'yaml';
import type { TrainingConfig } from '@/lib/types';

export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  const uiConfig = body.config as TrainingConfig;
  const datasetName = body.datasetName as string | undefined;
  const runName = body.name as string | undefined;
  const gpuMode = body.gpuMode as string | undefined;
  const gpuIds = body.gpuIds as string | undefined;

  if (!datasetName) {
    return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
  }

  const outputDir: string = uiConfig.outputDir || '/tmp/ltx-training';
  try {
    fs.mkdirSync(outputDir, { recursive: true });
  } catch (err) {
    return NextResponse.json({ error: `Failed to create output directory: ${toErrorMessage(err)}` }, { status: 500 });
  }

  let preprocessedDataRoot: string | null = null;
  const dataset = db.select().from(trainingDatasets).where(eq(trainingDatasets.name, datasetName)).get();
  if (dataset) {
    preprocessedDataRoot = path.join(dataset.path, '.precomputed');
  }
  if (!preprocessedDataRoot && uiConfig.data?.preprocessedDataRoot) {
    preprocessedDataRoot = uiConfig.data.preprocessedDataRoot;
  }

  const yamlConfig = buildYamlConfig(uiConfig, preprocessedDataRoot);
  const configPath = path.join(outputDir, 'training_config.yaml');
  try {
    fs.writeFileSync(configPath, YAML.stringify(yamlConfig), 'utf-8');
  } catch (err) {
    return NextResponse.json({ error: `Failed to write config file: ${toErrorMessage(err)}` }, { status: 500 });
  }

  const result = db
    .insert(jobs)
    .values({
      type: 'training',
      name: runName || path.basename(outputDir),
      status: 'queued',
      config: JSON.stringify({
        ...uiConfig,
        configPath,
        gpuMode,
        gpuIds,
        datasetName,
        preprocessedDataRoot,
      }),
      queuePosition: nextQueuePosition(),
      logFile: path.join(outputDir, 'train.log'),
    })
    .returning()
    .get();

  return NextResponse.json({ ...result, config: parseJobConfig(result.config) ?? {} }, { status: 201 });
}

function nonEmptyOrNull<T>(arr: T[]): T[] | null {
  return arr.length > 0 ? arr : null;
}

function buildYamlConfig(uiConfig: TrainingConfig, preprocessedDataRoot: string | null): Record<string, unknown> {
  return {
    model: {
      model_path: uiConfig.model.modelPath || null,
      text_encoder_path: uiConfig.model.textEncoderPath || null,
      training_mode: uiConfig.model.trainingMode || 'lora',
      load_checkpoint: uiConfig.model.loadCheckpoint || null,
    },
    lora:
      uiConfig.model.trainingMode !== 'full'
        ? {
            rank: uiConfig.lora.rank,
            alpha: uiConfig.lora.alpha,
            dropout: uiConfig.lora.dropout,
            target_modules: uiConfig.lora.targetModules,
          }
        : undefined,
    training_strategy: {
      name: uiConfig.trainingStrategy.name,
      first_frame_conditioning_p: uiConfig.trainingStrategy.firstFrameConditioningP,
      with_audio: uiConfig.trainingStrategy.withAudio,
      audio_latents_dir: uiConfig.trainingStrategy.audioLatentsDir,
      h_flip: uiConfig.trainingStrategy.hFlip,
      first_frame_conditioning_noise: uiConfig.trainingStrategy.firstFrameConditioningNoise,
      temporal_boundary_loss_weight: uiConfig.trainingStrategy.temporalBoundaryLossWeight,
      temporal_boundary_frames: uiConfig.trainingStrategy.temporalBoundaryFrames,
    },
    optimization: {
      learning_rate: uiConfig.optimization.learningRate,
      steps: uiConfig.optimization.steps,
      batch_size: uiConfig.optimization.batchSize,
      gradient_accumulation_steps: uiConfig.optimization.gradientAccumulationSteps,
      max_grad_norm: uiConfig.optimization.maxGradNorm,
      optimizer_type: uiConfig.optimization.optimizerType,
      optimizer_params: {
        weight_decay: uiConfig.optimization.weightDecay,
        ...(uiConfig.optimization.optimizerType === 'muon' ? { adjust_lr_fn: 'match_rms_adamw' } : {}),
      },
      scheduler_type: uiConfig.optimization.schedulerType,
      scheduler_params: {
        num_warmup_steps: uiConfig.optimization.numWarmupSteps,
      },
      enable_gradient_checkpointing: uiConfig.optimization.enableGradientCheckpointing,
      weight_noise: {
        mode: uiConfig.optimization.weightNoise.mode,
        sigma: uiConfig.optimization.weightNoise.sigma,
      },
    },
    acceleration: {
      mixed_precision_mode: 'bf16',
    },
    data: {
      preprocessed_data_root: preprocessedDataRoot || '',
      num_dataloader_workers: 4,
    },
    validation: {
      prompts: uiConfig.validation.prompts.filter(Boolean),
      images: nonEmptyOrNull(uiConfig.validation.images.filter(Boolean)),
      negative_prompt: uiConfig.validation.negativePrompt,
      video_dims: uiConfig.validation.videoDims,
      frame_rate: uiConfig.validation.frameRate,
      seed: uiConfig.validation.seed,
      inference_steps: uiConfig.validation.inferenceSteps,
      interval: uiConfig.validation.interval,
      videos_per_prompt: uiConfig.validation.videosPerPrompt,
      guidance_scale: uiConfig.validation.guidanceScale,
      stg_scale: 0.0,
      stg_blocks: null,
      stg_mode: 'stg_av',
      generate_audio: uiConfig.validation.generateAudio,
      skip_initial_validation: uiConfig.validation.skipInitialValidation,
    },
    checkpoints: {
      interval: uiConfig.checkpoints.interval,
      keep_last_n: uiConfig.checkpoints.keepLastN,
      precision: uiConfig.checkpoints.precision,
      save_training_state: uiConfig.checkpoints.saveTrainingState,
      no_resume: true,
    },
    flow_matching: {
      timestep_sampling_mode: uiConfig.flowMatching.timestepSamplingMode,
      timestep_sampling_params: {},
      timestep_loss_weighting: uiConfig.flowMatching.timestepLossWeighting,
    },
    tensorboard: {
      enabled: true,
    },
    hub: {
      push_to_hub: false,
      hub_model_id: null,
    },
    output_dir: uiConfig.outputDir,
    seed: uiConfig.seed ?? 42,
  };
}
