import pytest
import torch

from ltx_trainer.validation_loss import ValidationLossAccumulator


def test_validation_loss_accumulator_computes_overall_and_sigma_means() -> None:
    accumulator = ValidationLossAccumulator()
    accumulator.update(
        losses=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        sigmas=torch.tensor([0.1, 0.3, 0.6, 0.9]),
    )

    metrics = accumulator.metrics()

    assert metrics["validation/loss"] == pytest.approx(2.5)
    assert metrics["validation/num_samples"] == 4
    assert metrics["validation/loss_sigma_0.00-0.25"] == pytest.approx(1.0)
    assert metrics["validation/loss_sigma_0.25-0.50"] == pytest.approx(2.0)
    assert metrics["validation/loss_sigma_0.50-0.75"] == pytest.approx(3.0)
    assert metrics["validation/loss_sigma_0.75-1.00"] == pytest.approx(4.0)


def test_validation_loss_accumulator_round_trips_reduction_tensor() -> None:
    accumulator = ValidationLossAccumulator()
    accumulator.update(
        losses=torch.tensor([0.25, 0.75]),
        sigmas=torch.tensor([0.2, 0.8]),
    )

    restored = ValidationLossAccumulator.from_tensor(accumulator.to_tensor(torch.device("cpu")))

    assert restored.metrics() == accumulator.metrics()


def test_validation_loss_accumulator_rejects_mismatched_batch_lengths() -> None:
    accumulator = ValidationLossAccumulator()

    with pytest.raises(ValueError, match="same number"):
        accumulator.update(torch.tensor([1.0]), torch.tensor([0.1, 0.2]))
