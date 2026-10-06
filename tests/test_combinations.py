"""Crossed route/KL choices, checked against independent differentiable formulas."""

import math

import pytest
import torch

from klpo import klpo_sequence_full_loss, klpo_sequence_loss, klpo_token_loss, klpo_sequence_topk_loss


@pytest.mark.parametrize('action', [0, 3])
@pytest.mark.parametrize('floor_case', ['none', 'trainer', 'sampler', 'both'])
def test_sequence_topk_matches_literal_squared_residual_including_tail_clamp(action, floor_case):
    p = [.5, .35, .1, .05] if floor_case in {'trainer', 'both'} else [.3, .25, .25, .2]
    q = [.6, .3, .06, .04] if floor_case in {'sampler', 'both'} else [.35, .3, .2, .15]
    logits = torch.tensor(p, dtype=torch.float64).log().repeat(2, 3, 1).requires_grad_()
    p_log = logits.log_softmax(-1)
    q_log = torch.tensor(q, dtype=torch.float64).log().repeat(2, 3, 1).requires_grad_()
    rewards = torch.tensor([1., -.3], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    beta, floor = .7, .2
    loss, stats = klpo_sequence_topk_loss(p_log[..., action], q_log[..., action], rewards, mask,
        conditional_log_probs=p_log[..., :2], behavior_conditional_log_probs=q_log[..., :2],
        beta=beta, tail_floor=floor)
    q_head, p_head = q_log.detach()[..., :2].exp(), p_log[..., :2].exp()
    qt, pt = (1 - q_head.sum(-1)).clamp_min(floor), (1 - p_head.sum(-1)).clamp_min(floor)
    kl = (q_head * (q_log.detach()[..., :2] - p_log[..., :2])).sum(-1) + qt * (qt.log() - pt.log())
    d = rewards.detach() - beta * ((p_log[..., action] - q_log.detach()[..., action] + kl) * mask).sum(-1)
    regression = d.square().mean() / (2 * beta)
    expected, = torch.autograd.grad(regression, logits, retain_graph=True)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(stats['regression_loss'], regression.detach())
    torch.testing.assert_close(stats['residual'], d.detach())
    assert stats['floored_tokens'].item() == (0 if floor_case == 'none' else mask.sum().item())
    assert q_log.grad is None and rewards.grad is None
    assert all(not value.requires_grad for value in stats.values())


@pytest.mark.parametrize('k,full_record', [(3, False), (4, True)])
def test_sequence_topk_exact_full_vocabulary_limits(k, full_record):
    logits = torch.tensor([[[.4, -.7, .2, -.2], [.2, -.3, .5, -.8]]], dtype=torch.float64, requires_grad=True)
    p_log = logits.log_softmax(-1)
    q_log = torch.tensor([[[.4, .3, .2, .1], [.5, .2, .2, .1]]], dtype=torch.float64).log()
    args = (p_log[..., -1], q_log[..., -1], torch.tensor([.8]), torch.ones(1, 2, dtype=torch.bool))
    topk, topk_stats = klpo_sequence_topk_loss(*args, conditional_log_probs=p_log[..., :k],
        behavior_conditional_log_probs=q_log[..., :k], full_vocabulary=full_record)
    full, full_stats = klpo_sequence_full_loss(*args, full_log_probs=p_log, behavior_full_log_probs=q_log)
    actual, = torch.autograd.grad(topk, logits, retain_graph=True)
    expected, = torch.autograd.grad(full, logits)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(topk_stats['regression_loss'], full_stats['regression_loss'])


@pytest.mark.parametrize('route', ['sequence', 'token'])
def test_singleton_sampled_action_head_matches_binary_partition(route):
    logits = torch.tensor([[[.4, -.7, .2, -.2], [.2, -.3, .5, -.8]]], dtype=torch.float64, requires_grad=True)
    p_log = logits.log_softmax(-1)
    q_log = torch.tensor([[[.4, .3, .2, .1], [.5, .2, .2, .1]]], dtype=torch.float64).log()
    args = (p_log[..., 0], q_log[..., 0], torch.tensor([.8]), torch.ones(1, 2, dtype=torch.bool))
    head = dict(conditional_log_probs=p_log[..., :1], behavior_conditional_log_probs=q_log[..., :1])
    if route == 'sequence':
        binary, binary_stats = klpo_sequence_loss(*args)
        topk, topk_stats = klpo_sequence_topk_loss(*args, **head)
        torch.testing.assert_close(binary_stats['residual'], topk_stats['residual'])
    else:
        binary, _ = klpo_token_loss(*args, kl_estimator='binary')
        topk, _ = klpo_token_loss(*args, kl_estimator='topk', **head)
    actual, = torch.autograd.grad(binary, logits, retain_graph=True)
    expected, = torch.autograd.grad(topk, logits)
    torch.testing.assert_close(actual, expected)


def test_token_binary_matches_derivative_of_logp_plus_binary_kl_with_token_feedback():
    logits = torch.tensor([[[.3, -.2, .1], [-.4, .8, .5]], [[.1, -.9, .4], [.2, -.7, .1]]],
                          dtype=torch.float64, requires_grad=True)
    full = logits.log_softmax(-1)
    current = full[..., 0]
    old = torch.tensor([[-2., -.9], [-.7, -1.]], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1., -.2], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    beta = .4
    loss, stats = klpo_token_loss(current, old, rewards, mask, beta=beta, kl_estimator='binary')
    q, p = old.detach().exp(), current.exp()
    binary_kl = q * (old.detach() - current) + (1 - q) * (torch.log1p(-q) - torch.log1p(-p))
    h = (rewards[:, None] - beta * (current - old)).detach()
    direct = -(h * (current + binary_kl) * mask).sum(-1).mean()
    expected, = torch.autograd.grad(direct, logits, retain_graph=True)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(stats['return_coefficient'], h.masked_fill(~mask, 0.))
    assert old.grad is None and rewards.grad is None
    assert all(not x.requires_grad for x in stats.values())
    assert 'residual' not in stats and 'regression_loss' not in stats


def test_token_binary_equals_full_kl_in_a_binary_vocabulary():
    logits = torch.tensor([[[.2, -.3], [-.7, .4]]], dtype=torch.float64, requires_grad=True)
    full = logits.log_softmax(-1)
    q_log = torch.tensor([[[.6, .4], [.3, .7]]], dtype=torch.float64).log()
    args = (full[..., 0], q_log[..., 0], torch.tensor([.8]), torch.ones(1, 2, dtype=torch.bool))
    binary, _ = klpo_token_loss(*args, kl_estimator='binary')
    exact, _ = klpo_token_loss(*args, kl_estimator='full',
        conditional_log_probs=full, behavior_conditional_log_probs=q_log)
    actual, = torch.autograd.grad(binary, logits, retain_graph=True)
    expected, = torch.autograd.grad(exact, logits)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_token_binary_precision_and_stale_sampler(dtype):
    values = [-1e-10, -1., -0.5] if dtype != torch.bfloat16 else [-.01, -1., -.5]
    current = torch.tensor([values], dtype=dtype, requires_grad=True)
    old = torch.tensor([[2 * values[0], -1000., -1.]], dtype=dtype, requires_grad=True)
    loss, stats = klpo_token_loss(current, old, torch.ones(1), torch.ones(1, 3, dtype=torch.bool),
                                           kl_estimator='binary')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    assert stats['return_coefficient'][0, 1] < -90
    assert stats['correction'][0, 0].item() == pytest.approx(2., rel=.01)
    assert old.grad is None


@pytest.mark.parametrize('route', ['sequence_topk', 'token_binary'])
def test_replay_updates_feedback_but_preserves_sampler(route):
    old = torch.tensor([[[.6, .3, .1], [.5, .2, .3]]], dtype=torch.float64).log()
    frozen = old.clone()
    outputs = []
    for shift in (0., .4):
        full = (old + torch.tensor([shift, 0., 0.])).log_softmax(-1)
        args = (full[..., 0], old[..., 0], torch.ones(1), torch.ones(1, 2, dtype=torch.bool))
        if route == 'sequence_topk':
            _, stats = klpo_sequence_topk_loss(*args, conditional_log_probs=full[..., :1], behavior_conditional_log_probs=old[..., :1])
            outputs.append(stats['residual'])
        else:
            _, stats = klpo_token_loss(*args, kl_estimator='binary')
            outputs.append(stats['return_coefficient'])
    assert not torch.equal(*outputs)
    torch.testing.assert_close(old, frozen)


@pytest.mark.parametrize('kwargs', [
    {'kl_estimator': 'invalid'}, {'kl_estimator': 'full'},
    {'kl_estimator': 'binary', 'full_vocabulary': True},
    {'kl_estimator': 'binary', 'conditional_log_probs': torch.ones(1, 2, 1)},
])
def test_sc_invalid_estimator_or_record_contract(kwargs):
    with pytest.raises(ValueError):
        klpo_token_loss(torch.full((1, 2), -1.), torch.full((1, 2), -2.),
                                 torch.ones(1), torch.ones(1, 2, dtype=torch.bool), **kwargs)


@pytest.mark.parametrize('floor', [0., -1., 1., math.inf, math.nan])
def test_sequence_topk_invalid_tail_floor(floor):
    with pytest.raises(ValueError, match='tail_floor'):
        klpo_sequence_topk_loss(torch.full((1, 2), -1.), torch.full((1, 2), -2.),
            torch.ones(1), torch.ones(1, 2, dtype=torch.bool),
            conditional_log_probs=torch.full((1, 2, 1), -1.),
            behavior_conditional_log_probs=torch.full((1, 2, 1), -2.), tail_floor=floor)


def test_token_binary_rejects_rounded_one_and_overflow():
    for value, error in ((0., ValueError), (-1e-40, FloatingPointError)):
        with pytest.raises(error):
            klpo_token_loss(torch.tensor([[value]]), torch.tensor([[-1.]]),
                                     torch.ones(1), torch.ones(1, 1, dtype=torch.bool), kl_estimator='binary')
