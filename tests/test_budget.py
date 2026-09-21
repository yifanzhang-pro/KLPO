import math

import pytest
import torch

from klpo import kl_budget_scale, predicted_kl


def test_curvature_matches_realized_kl_for_a_small_displacement():
    logits = torch.tensor([[.1, -.7, .9], [-.1, .3, .8]], dtype=torch.float64)
    direction = torch.tensor([[.2, -.1, .6], [.5, -.2, .1]], dtype=torch.float64)
    mask, temperature = torch.ones(2, dtype=torch.bool), .7
    q = predicted_kl(logits, direction, mask, temperature=temperature)
    eps = 1e-4
    old = (logits / temperature).log_softmax(-1)
    new = ((logits + eps * direction) / temperature).log_softmax(-1)
    actual = (old.exp() * (old - new)).sum(-1).mean()
    torch.testing.assert_close(actual / eps ** 2, q, rtol=1e-4, atol=1e-7)
    alpha = kl_budget_scale(q, .001, alpha_max=.8)
    assert alpha <= .8 and alpha ** 2 * q <= .001 + 1e-12


@pytest.mark.parametrize('q', [None, 0., -1., math.nan, math.inf])
def test_invalid_or_zero_probe_skips(q):
    assert kl_budget_scale(q, .1) == 0
    assert kl_budget_scale(q, None) == 1


def test_zero_budget_and_padding():
    assert kl_budget_scale(1., 0.) == 0
    logits = torch.tensor([[.1, .2], [math.nan, math.inf]])
    jvp = torch.tensor([[.1, -.3], [math.nan, math.inf]])
    q = predicted_kl(logits, jvp, torch.tensor([True, False]))
    assert torch.isfinite(q)
    assert math.isnan(predicted_kl(logits, jvp, torch.tensor([False, False])))
    assert kl_budget_scale(100., .01) == .01


@pytest.mark.parametrize('budget,cap', [(-1., 1.), (math.inf, 1.), (.1, 0.), (.1, 1.1)])
def test_invalid_configuration(budget, cap):
    with pytest.raises(ValueError):
        kl_budget_scale(1., budget, alpha_max=cap)
