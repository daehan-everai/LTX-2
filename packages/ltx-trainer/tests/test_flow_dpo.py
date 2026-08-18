import torch

from ltx_trainer.training_strategies.flow_dpo import KLING_FLOW_DPO_BETA, flow_dpo_loss_from_errors, masked_mse


def test_flow_dpo_loss_falls_when_policy_prefers_chosen() -> None:
    model_w_err = torch.tensor([0.4, 0.5])
    model_l_err = torch.tensor([1.2, 1.3])
    ref_w_err = torch.tensor([0.8, 0.8])
    ref_l_err = torch.tensor([0.8, 0.8])

    aligned = flow_dpo_loss_from_errors(model_w_err, model_l_err, ref_w_err, ref_l_err)
    inverted = flow_dpo_loss_from_errors(model_l_err, model_w_err, ref_w_err, ref_l_err)

    assert aligned.mean().item() < inverted.mean().item()
    assert torch.all(aligned < inverted)


def test_flow_dpo_loss_uses_kling_constant_beta() -> None:
    ones = torch.ones(2)
    zeros = torch.zeros(2)
    loss = flow_dpo_loss_from_errors(zeros, ones, ones, zeros, beta=KLING_FLOW_DPO_BETA)
    w_diff = zeros - ones
    l_diff = ones - zeros
    inside = -0.5 * KLING_FLOW_DPO_BETA * (w_diff - l_diff)
    expected = torch.nn.functional.softplus(-inside)
    torch.testing.assert_close(loss, expected)


def test_masked_mse_ignores_conditioning_tokens() -> None:
    pred = torch.zeros(1, 4, 2)
    target = torch.ones(1, 4, 2)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    pred[:, 2:] = 3.0
    err = masked_mse(pred, target, mask)
    assert err.shape == (1,)
    assert err.item() == 4.0
