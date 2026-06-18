import re
from pathlib import Path

from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from torch import nn


def detect_checkpoint_lora_modules(checkpoint_path: Path) -> set[str]:
    """Read safetensors metadata only (no tensor data) and return the set of
    module paths that have LoRA weights in the checkpoint.

    Args:
        checkpoint_path: Path to a LoRA checkpoint in .safetensors format.
    Returns:
        Set of module paths (e.g. "transformer_blocks.0.attn1.to_k") that have
        LoRA adapter weights in the checkpoint.
    """
    paths: set[str] = set()
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            key = key.replace("diffusion_model.", "", 1)
            m = re.match(r"^(.+?)\.lora_[AB]\.", key)
            if m:
                paths.add(m.group(1))
    return paths


def matches_any_lora_target(module_name: str, target_modules: list[str]) -> bool:
    """Mirror PEFT's list-mode target matching: exact match or suffix match
    (e.g. "to_k" matches "x.y.to_k")."""
    return module_name in target_modules or any(module_name.endswith(f".{t}") for t in target_modules)


def freeze_extra_lora_layers(peft_model: nn.Module, train_target_modules: list[str]) -> int:
    """Freeze LoRA adapter params for every BaseTunerLayer whose name does NOT
    match any training target.

    Args:
        peft_model: The PEFT-wrapped model.
        train_target_modules: The target modules that should remain trainable.
    Returns:
        The number of LoRA layers that were frozen.
    """
    frozen = 0
    for name, module in peft_model.named_modules():
        if isinstance(module, BaseTunerLayer) and not matches_any_lora_target(name, train_target_modules):
            for attr in module.adapter_layer_names:
                d = getattr(module, attr, None)
                if d:
                    for sub in d.values():
                        sub.requires_grad_(False)
            frozen += 1
    return frozen
