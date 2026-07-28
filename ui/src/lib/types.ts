export interface SourceFolder {
  id: number;
  path: string;
  name: string;
  mediaType: 'images' | 'videos' | 'mixed' | 'audio';
  fileCount: number;
  createdAt: string;
}

export interface ProcessingJob {
  id: number;
  type: 'preprocess' | 'merge' | 'training';
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  config: Record<string, unknown>;
  queuePosition: number;
  pid: number | null;
  logFile: string | null;
  stopRequested: boolean;
  progress: number | null;
  error: string | null;
  startedAt: string | null;
  completedAt: string | null;
  createdAt: string;
  /** Present only for completed preprocess jobs: whether the .precomputed output directory exists on disk */
  outputExists?: boolean;
}

export interface TrainingDataset {
  id: number;
  name: string;
  path: string;
  buckets: DatasetBucket[];
  pathExists: boolean;
  buildStatus: 'queued' | 'running' | null;
  createdAt: string;
}

export interface DatasetBucket {
  folderName: string;
  folderPath: string;
  jobId: number;
  resolution: number;
  frameCount: number;
  bucketKeys: string[];
  hasAudio: boolean;
  hasHFlip: boolean;
  isAudioOnly?: boolean;
}

export interface TrainingConfig {
  model: {
    modelPath: string;
    textEncoderPath: string;
    trainingMode: 'lora' | 'full';
    loadCheckpoint: string | null;
  };
  lora: {
    rank: number;
    alpha: number;
    dropout: number;
    targetModules: string[];
    freezeExtraModules: boolean;
  };
  trainingStrategy: {
    name: 'text_to_video' | 'video_to_video';
    firstFrameConditioningP: number;
    // text_to_video only
    withAudio: boolean;
    audioLatentsDir: string;
    hFlip: boolean;
    // shared
    temporalBoundaryLossWeight: number;
    temporalBoundaryFrames: number;
    captionDropoutP: number;
  };
  optimization: {
    learningRate: number;
    steps: number;
    batchSize: number;
    gradientAccumulationSteps: number;
    maxGradNorm: number;
    optimizerType: string;
    weightDecay: number;
    optimizerParams?: Record<string, unknown>;
    schedulerType: string;
    numWarmupSteps: number;
    enableGradientCheckpointing: boolean;
    gradientCheckpointingRatio: number;
    weightNoise: {
      mode: 'none' | 'relative' | 'absolute';
      sigma: number;
    };
  };
  flowMatching: {
    timestepSamplingMode: 'uniform' | 'shifted_logit_normal';
    timestepLossWeighting: 'none' | 'bell' | 'weighted';
  };
  data?: {
    preprocessedDataRoot?: string;
  };
  validation: {
    prompts: string[];
    images: string[];
    referenceVideos: string[];
    referenceDownscaleFactor: number;
    includeReferenceInOutput: boolean;
    negativePrompt: string;
    videoDims: [number, number, number];
    frameRate: number;
    seed: number;
    inferenceSteps: number;
    interval: number;
    videosPerPrompt: number;
    guidanceScale: number;
    generateAudio: boolean;
    skipInitialValidation: boolean;
  };
  checkpoints: {
    interval: number;
    keepLastN: number;
    precision: string;
    saveTrainingState: 'full' | 'minimal' | 'off';
    resume: boolean;
  };
  outputDir: string;
  seed: number;
}

export interface AppSettings {
  modelPath: string;
  textEncoderPath: string;
  outputDir: string;
  datasetDir: string;
  scriptsDir: string;
}

export type ResolutionOption = 512 | 768 | 1024 | 1280 | 1440;
export type FrameCountOption = 1 | 25 | 49 | 73 | 121 | 145;

export const RESOLUTION_OPTIONS: ResolutionOption[] = [512, 768, 1024, 1280, 1440];
export const FRAME_COUNT_OPTIONS: FrameCountOption[] = [1, 25, 49, 73, 121, 145];
