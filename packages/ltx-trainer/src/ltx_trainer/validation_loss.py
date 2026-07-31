"""Deterministic aggregation helpers for holdout validation loss."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import torch
from torch import Tensor


DEFAULT_SIGMA_BOUNDARIES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class ValidationLossAccumulator:
    """Accumulate loss totals and sigma-bucket totals across validation batches."""

    boundaries: tuple[float, ...] = DEFAULT_SIGMA_BOUNDARIES
    loss_sum: float = 0.0
    sample_count: int = 0
    bucket_loss_sums: list[float] = field(default_factory=list)
    bucket_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.boundaries) < 2:
            raise ValueError("boundaries must have at least two values")
        if any(left >= right for left, right in zip(self.boundaries[:-1], self.boundaries[1:], strict=True)):
            raise ValueError("boundaries must be strictly increasing")
        num_buckets = len(self.boundaries) - 1
        if not self.bucket_loss_sums:
            self.bucket_loss_sums = [0.0] * num_buckets
        if not self.bucket_counts:
            self.bucket_counts = [0] * num_buckets

    def update(self, losses: Tensor, sigmas: Tensor) -> None:
        """Add one batch of per-sample losses and sigmas."""
        loss_values = losses.detach().float().cpu().tolist()
        sigma_values = sigmas.detach().float().cpu().tolist()
        if len(loss_values) != len(sigma_values):
            raise ValueError("losses and sigmas must contain the same number of samples")

        self.loss_sum += sum(loss_values)
        self.sample_count += len(loss_values)
        for loss, sigma in zip(loss_values, sigma_values, strict=True):
            bucket = bisect.bisect_right(self.boundaries, sigma) - 1
            bucket = max(0, min(bucket, len(self.bucket_counts) - 1))
            self.bucket_loss_sums[bucket] += loss
            self.bucket_counts[bucket] += 1

    def to_tensor(self, device: torch.device) -> Tensor:
        """Serialize sums and counts for distributed reduction."""
        values: list[float] = [self.loss_sum, float(self.sample_count)]
        for loss_sum, count in zip(self.bucket_loss_sums, self.bucket_counts, strict=True):
            values.extend((loss_sum, float(count)))
        return torch.tensor(values, dtype=torch.float64, device=device)

    @classmethod
    def from_tensor(
        cls,
        values: Tensor,
        boundaries: tuple[float, ...] = DEFAULT_SIGMA_BOUNDARIES,
    ) -> ValidationLossAccumulator:
        """Restore an accumulator from a reduced tensor."""
        raw = values.detach().double().cpu().tolist()
        num_buckets = len(boundaries) - 1
        expected = 2 + 2 * num_buckets
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} values, got {len(raw)}")
        return cls(
            boundaries=boundaries,
            loss_sum=raw[0],
            sample_count=round(raw[1]),
            bucket_loss_sums=[raw[2 + 2 * index] for index in range(num_buckets)],
            bucket_counts=[round(raw[3 + 2 * index]) for index in range(num_buckets)],
        )

    def metrics(self, prefix: str = "validation") -> dict[str, float]:
        """Return overall and populated sigma-bucket means."""
        if self.sample_count == 0:
            raise ValueError("Cannot compute validation metrics without samples")
        metrics = {
            f"{prefix}/loss": self.loss_sum / self.sample_count,
            f"{prefix}/num_samples": float(self.sample_count),
        }
        for index, (loss_sum, count) in enumerate(zip(self.bucket_loss_sums, self.bucket_counts, strict=True)):
            if count == 0:
                continue
            left = self.boundaries[index]
            right = self.boundaries[index + 1]
            metrics[f"{prefix}/loss_sigma_{left:.2f}-{right:.2f}"] = loss_sum / count
        return metrics
