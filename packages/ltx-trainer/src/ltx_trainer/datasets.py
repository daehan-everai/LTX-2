from typing import Any
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from ltx_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        data_sources: dict[str, str] | list[str] | None = None,
        h_flip: bool = False,
        optional_sources: set[str] | None = None,
    ) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.

        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
            h_flip: Whether to enable horizontal flip augmentation. When True, each sample
                has a 50% chance of loading from ``latents_h_flip/`` instead of ``latents/``.
                Requires the dataset to be preprocessed with ``--with-h-flip``.
            optional_sources: Output keys of sources whose files may be missing per-sample.
                Affected samples are kept; ``__getitem__`` returns a sentinel dict
                (``has_audio``/``has_ref`` = False) for the missing source instead.
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode with optional references (mixed dataset)
            dataset = PrecomputedDataset(
                "data/",
                {"latents": "latents", "conditions": "conditions", "reference_latents": "ref_latents"},
                optional_sources={"ref_latents"},
            )
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.optional_sources = set(optional_sources or ())
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()
        self.h_flip = h_flip
        self.h_flip_latents_path = self._discover_h_flip_latents()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    @staticmethod
    def _is_audio_source(output_key: str) -> bool:
        """Return True for sources whose output key contains ``"audio"``."""
        return "audio" in output_key.lower()

    def _is_optional_source(self, output_key: str) -> bool:
        """Optional sources may have missing files without excluding the sample."""
        return output_key in self.optional_sources

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            if not source_path.exists():
                if self._is_optional_source(output_key):
                    logger.warning(f"Optional source directory '{dir_name}' does not exist at {source_path}.")
                else:
                    raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path | None]]:
        """Discover all valid sample files across all data sources.

        Required sources must have a matching file for each sample. Optional sources
        store ``None`` when the file is missing.
        """
        # Use first data source as the reference to discover samples
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]
        data_files = sorted(list(data_path.glob("**/*.pt")))

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        # Initialize sample files dict
        sample_files: dict[str, list[Path | None]] = {output_key: [] for output_key in self.data_sources.values()}

        # For each data file, find corresponding files in required sources
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            if self._all_required_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _all_required_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        """Check that every required source has a matching file for this sample."""
        for dir_name, output_key in self.data_sources.items():
            if self._is_optional_source(output_key):
                continue
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                logger.warning(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False

        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(
        self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path | None]]
    ) -> None:
        """Add a valid sample to the sample_files tracking.

        Required sources are known to exist at this point; optional sources are
        checked individually since individual files may be absent.
        """
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if self._is_optional_source(output_key):
                path = expected_path.relative_to(self.source_paths[dir_name]) if expected_path.exists() else None
            else:
                path = expected_path.relative_to(self.source_paths[dir_name])
            sample_files[output_key].append(path)

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        # Verify all output keys have the same number of samples
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def _discover_h_flip_latents(self) -> Path | None:
        """Discover the ``latents_h_flip/`` sibling directory for horizontal flip augmentation.

        Returns the resolved Path if the directory exists, or None if h_flip is disabled.
        Raises FileNotFoundError if h_flip is enabled but the directory is missing.

        Individual samples without a flipped counterpart file are handled gracefully
        at load time by falling back to the original latent.
        """
        h_flip_dir = self.data_root / "latents_h_flip"

        if not self.h_flip:
            return None

        if not h_flip_dir.exists():
            raise FileNotFoundError(
                f"h_flip is enabled but '{h_flip_dir}' does not exist. "
                f"Re-run dataset preprocessing with --with-h-flip to generate flipped latents."
            )

        logger.info(f"H-flip augmentation enabled, using flipped latents from {h_flip_dir}")
        return h_flip_dir

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Decide whether to use horizontally flipped latents (50% chance)
        use_h_flip = self.h_flip_latents_path is not None and torch.rand(1).item() < 0.5

        result = {}

        for dir_name, output_key in self.data_sources.items():
            file_rel_path = self.sample_files[output_key][index]

            # Optional source file missing for this sample — return sentinel
            if file_rel_path is None:
                flag_key = "has_audio" if self._is_audio_source(output_key) else "has_ref"
                result[output_key] = {flag_key: torch.tensor(False)}
                continue

            source_path = self.source_paths[dir_name]

            file_path = source_path / file_rel_path
            if use_h_flip and dir_name == "latents":
                h_flip_path = self.h_flip_latents_path / file_rel_path
                if h_flip_path.exists():
                    file_path = h_flip_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format (video sources only — audio is already 3-D)
                if "latent" in dir_name.lower() and not self._is_audio_source(output_key):
                    data = self._normalize_video_latents(data)

                if self._is_optional_source(output_key):
                    flag_key = "has_audio" if self._is_audio_source(output_key) else "has_ref"
                    data[flag_key] = torch.tensor(True)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

            # Track the latents filename for logging in the trainer
            if dir_name == "latents":
                result["_sample_path"] = str(file_rel_path)

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data


def _collate_audio_source(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of per-sample audio dicts.

    Samples with audio contain ``{"has_audio": True, "latents": Tensor, ...}``.
    Samples without audio contain only ``{"has_audio": False}``.

    When at least one sample has audio, absent-audio rows are zero-padded to the
    maximum latent shape found among present samples so the batch can be stacked.
    """
    has_audio_flags = [s["has_audio"].item() for s in samples]
    has_audio = torch.tensor(has_audio_flags, dtype=torch.bool)

    present = [s for s in samples if s["has_audio"].item()]

    if not present:
        return {"has_audio": has_audio}

    # Use first present sample as shape/type reference
    ref = present[0]["latents"]

    # Find maximum size along each dimension to pad to
    shapes = [s["latents"].shape for s in present]
    max_shape = list(shapes[0])
    for shape in shapes[1:]:
        for dim in range(len(max_shape)):
            max_shape[dim] = max(max_shape[dim], shape[dim])

    def _pad_to(tensor: Tensor, target_shape: list[int]) -> Tensor:
        if list(tensor.shape) == target_shape:
            return tensor
        padded = torch.zeros(target_shape, dtype=tensor.dtype)
        slices = tuple(slice(0, s) for s in tensor.shape)
        padded[slices] = tensor
        return padded

    latents_list = []
    for sample in samples:
        if sample["has_audio"].item():
            latents_list.append(_pad_to(sample["latents"], max_shape))
        else:
            latents_list.append(torch.zeros(max_shape, dtype=ref.dtype))

    result: dict[str, Any] = {
        "has_audio": has_audio,
        "latents": torch.stack(latents_list),
    }

    # Collate scalar metadata from present samples (use first present sample's values for absent ones)
    ref_sample = present[0]
    for key in ref_sample:
        if key in ("has_audio", "latents"):
            continue
        values = [s.get(key, ref_sample[key]) for s in samples]
        try:
            result[key] = default_collate(values)
        except (TypeError, RuntimeError):
            result[key] = values

    return result


def _collate_ref_source(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of per-sample reference dicts.

    Unlike audio, reference presence changes the transformer sequence length, so a
    batch cannot mix referenced and unreferenced samples. Homogeneous batches are
    collated normally; mixed batches raise with guidance to use batch_size=1.
    """
    has_ref_flags = [s["has_ref"].item() for s in samples]
    has_ref = torch.tensor(has_ref_flags, dtype=torch.bool)

    if not has_ref.any():
        return {"has_ref": has_ref}

    if not has_ref.all():
        raise ValueError(
            "A batch mixes samples with and without reference latents, which is not supported "
            "because reference conditioning changes the sequence length. "
            "Use optimization.batch_size: 1 when training on a mixed dataset."
        )

    return default_collate(samples)


def collate_with_optional_sources(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that handles optional sources whose files may be missing.

    Keys whose batch values are dicts containing a ``has_audio`` field are
    collated with :func:`_collate_audio_source`; dicts containing a ``has_ref``
    field with :func:`_collate_ref_source`. All other keys use the standard
    PyTorch collation.
    """
    all_keys = batch[0].keys()
    result: dict[str, Any] = {}

    audio_keys: set[str] = set()
    ref_keys: set[str] = set()
    regular_keys: set[str] = set()

    for key in all_keys:
        sample_val = batch[0][key]
        if isinstance(sample_val, dict) and "has_audio" in sample_val:
            audio_keys.add(key)
        elif isinstance(sample_val, dict) and "has_ref" in sample_val:
            ref_keys.add(key)
        else:
            regular_keys.add(key)

    for key in audio_keys:
        result[key] = _collate_audio_source([s[key] for s in batch])

    for key in ref_keys:
        result[key] = _collate_ref_source([s[key] for s in batch])

    if regular_keys:
        regular_batch = [{k: s[k] for k in regular_keys} for s in batch]
        result.update(default_collate(regular_batch))

    return result
