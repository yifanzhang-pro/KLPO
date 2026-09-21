import math

import pytest
import torch

from klpo import klpo_sequence_full_loss, klpo_sequence_loss, klpo_token_loss, klpo_sequence_topk_loss


def test_binary_gradient_equals_squared_residual_not_importance_sampling():
    current = torch.tensor([[-1.2, -0.4, -2.], [-0.6, -2.3, -1.]], dtype=torch.float64, requires_grad=True)
    behavior = torch.tensor([[-2., -0.7, -1.], [-1.1, -0.9, -2.]], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    beta = 0.3
    loss, stats = klpo_sequence_loss(current, behavior, rewards, mask, beta=beta)
    p, q = current.exp(), behavior.detach().exp()
    k = q * (behavior.detach() - current) + (1 - q) * (torch.log1p(-q) - torch.log1p(-p))
    d = rewards.detach() - beta * ((current - behavior.detach() + k) * mask).sum(-1)
    direct = (d.square() / (2 * beta)).mean()
    expected, = torch.autograd.grad(direct, current)
    loss.backward()
    torch.testing.assert_close(current.grad, expected)
    torch.testing.assert_close(stats['residual'], d.detach())
    torch.testing.assert_close(stats['regression_loss'], direct.detach())
    torch.testing.assert_close(stats['correction'][mask], ((1 - q) / (1 - p))[mask].detach())
    assert behavior.grad is None and rewards.grad is None
    assert all(not x.requires_grad for x in stats.values())


def test_single_rollout_and_constant_rewards_keep_learning_signal():
    for batch in (1, 3):
        logps = torch.full((batch, 2), -1., requires_grad=True)
        loss, _ = klpo_sequence_loss(logps, logps.detach(), torch.ones(batch), torch.ones_like(logps, dtype=torch.bool))
        loss.backward()
        torch.testing.assert_close(logps.grad, torch.full_like(logps, -1 / batch))


def test_token_sum_not_length_average():
    current = torch.full((2, 5), -1., requires_grad=True)
    mask = torch.tensor([[True, False, False, False, False], [True] * 5])
    loss, _ = klpo_sequence_loss(current, current.detach(), torch.ones(2), mask)
    loss.backward()
    torch.testing.assert_close(current.grad.sum(-1), torch.tensor([-.5, -2.5]))


@pytest.mark.parametrize('route', ['binary', 'full', 'sequence_topk', 'token_binary', 'token_full', 'token_topk'])
def test_padding_nan_inf_and_detached_records(route):
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    full = logits.log_softmax(-1)
    old = torch.randn_like(logits).log_softmax(-1).detach().requires_grad_()
    mask = torch.tensor([[True, False, False], [True, False, True]])
    sampled = full[..., 0].masked_fill(~mask, float('nan'))
    sampled_old = old[..., 0].masked_fill(~mask, float('inf'))
    rewards = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
    args = (sampled, sampled_old, rewards, mask)
    if route == 'binary':
        loss, stats = klpo_sequence_loss(*args)
    elif route == 'token_binary':
        loss, stats = klpo_token_loss(*args, kl_estimator='binary')
    elif route == 'full':
        loss, stats = klpo_sequence_full_loss(*args,
            full_log_probs=full.masked_fill(~mask[..., None], float('nan')),
            behavior_full_log_probs=old.masked_fill(~mask[..., None], float('inf')))
    else:
        k = 4 if route == 'token_full' else 2
        loss_fn = klpo_sequence_topk_loss if route == 'sequence_topk' else klpo_token_loss
        options = {} if route == 'sequence_topk' else {'kl_estimator': 'topk'}
        loss, stats = loss_fn(*args, **options,
            conditional_log_probs=full[..., :k].masked_fill(~mask[..., None], float('nan')),
            behavior_conditional_log_probs=old[..., :k].masked_fill(~mask[..., None], float('inf')),
            full_vocabulary=route == 'token_full')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[~mask].eq(0).all()
    assert old.grad is None and rewards.grad is None
    assert all(not x.requires_grad for x in stats.values())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_binary_near_one_uses_log_complements(dtype):
    current = torch.tensor([[-1e-10]], dtype=dtype, requires_grad=True)
    behavior = torch.tensor([[-2e-10]], dtype=dtype)
    loss, stats = klpo_sequence_loss(current, behavior, torch.ones(1), torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    assert stats['correction'].item() == pytest.approx(2., rel=1e-5)
    assert current.grad.item() == pytest.approx(-2., rel=1e-5)
    assert torch.isfinite(loss)


def test_extremely_stale_behavior_does_not_exponentiate_action_ratio():
    current = torch.tensor([[-1.]], requires_grad=True)
    old = torch.tensor([[-1000.]])
    loss, stats = klpo_sequence_loss(current, old, torch.ones(1), torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    assert stats['residual'].abs().item() > 90


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_promoted(dtype):
    current = torch.full((2, 3), -1., dtype=dtype, requires_grad=True)
    loss, stats = klpo_sequence_loss(current, current.detach() - .1, torch.ones(2), torch.ones(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.dtype == torch.float32 and stats['residual'].dtype == torch.float32
    assert torch.isfinite(current.grad).all()


def test_reuse_recomputes_dynamic_residual_without_changing_sampler():
    old = torch.full((1, 2), -1.)
    frozen = old.clone()
    residuals = []
    for value in (-1., -0.8):
        _, stats = klpo_sequence_loss(torch.full_like(old, value), old, torch.ones(1), torch.ones_like(old, dtype=torch.bool))
        residuals.append(stats['residual'])
    assert not torch.equal(*residuals)
    torch.testing.assert_close(old, frozen)


def test_full_kl_keeps_kl_derivative():
    logits = torch.tensor([[[.5, -.3, 1.], [.2, -.1, -.7]]], dtype=torch.float64, requires_grad=True)
    full = logits.log_softmax(-1)
    old = torch.tensor([[[.2, .5, .3], [.6, .1, .3]]], dtype=torch.float64).log().requires_grad_()
    sampled, sampled_old = full[..., 1], old[..., 1]
    reward, mask, beta = torch.tensor([.7]), torch.ones(1, 2, dtype=torch.bool), .2
    loss, stats = klpo_sequence_full_loss(sampled, sampled_old, reward, mask,
        full_log_probs=full, behavior_full_log_probs=old, beta=beta)
    k = (old.detach().exp() * (old.detach() - full)).sum(-1)
    residual = reward - beta * (sampled - sampled_old.detach() + k).sum(-1)
    direct = residual.square().mean() / (2 * beta)
    expected, = torch.autograd.grad(direct, logits, retain_graph=True)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(stats['regression_loss'], direct.detach())
    assert old.grad is None


def test_zero_sampler_support_full_routes():
    logits = torch.tensor([[[.3, -.5, .1]]], dtype=torch.float64, requires_grad=True)
    full = logits.log_softmax(-1)
    old = torch.tensor([[[.3, .7, 0.]]], dtype=torch.float64).log()
    args = (full[..., 0], old[..., 0], torch.ones(1), torch.ones(1, 1, dtype=torch.bool))
    loss, _ = klpo_sequence_full_loss(*args, full_log_probs=full, behavior_full_log_probs=old)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('full_record', [True, False])
@pytest.mark.parametrize('action', [0, 3])
def test_score_centering_gradient_against_tail_distribution(full_record, action):
    logits = torch.tensor([[[.3, -.2, .1, .8]]], dtype=torch.float64, requires_grad=True)
    full = logits.log_softmax(-1)
    q = torch.tensor([[[.4, .3, .2, .1]]], dtype=torch.float64)
    k, beta = (4 if full_record else 2), .3
    head = q[..., :k].log().requires_grad_()
    old = q[..., action].log().requires_grad_()
    reward = torch.tensor([.8], dtype=torch.float64, requires_grad=True)
    loss, stats = klpo_token_loss(full[..., action], old, reward, torch.ones(1, 1, dtype=torch.bool),
        kl_estimator='topk', conditional_log_probs=full[..., :k], behavior_conditional_log_probs=head,
        full_vocabulary=full_record, beta=beta)
    with torch.no_grad():
        q_approx = q.clone()
        if not full_record:
            p = full.exp()
            q_approx[..., k:] = p[..., k:] * ((1 - q[..., :k].sum(-1)) / (1 - p[..., :k].sum(-1)))[..., None]
        score = -q_approx
        score[..., action] += 1
        expected = -stats['return_coefficient'][..., None] * score
    loss.backward()
    torch.testing.assert_close(logits.grad, expected)
    assert head.grad is None and old.grad is None and reward.grad is None


def test_topk_floor_and_full_vocabulary_no_zero_over_zero():
    full = torch.tensor([[[0., -100.]]], requires_grad=True)
    old = torch.tensor([[[-1e-8, -18.420681]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    for exact in (True, False):
        k = 2 if exact else 1
        loss, stats = klpo_token_loss(full[..., 0], old[..., 0], torch.ones(1), mask,
            kl_estimator='topk', conditional_log_probs=full[..., :k], behavior_conditional_log_probs=old[..., :k], full_vocabulary=exact)
        assert torch.isfinite(loss)
        assert stats['floored_tokens'].item() == (0 if exact else 1)


def test_topk_kl_derivative_supplies_score_centering_correction():
    logits = torch.tensor([[[.4, -.2, .8, -.5]]], dtype=torch.float64, requires_grad=True)
    p_log = logits.log_softmax(-1)
    q = torch.tensor([[[.4, .3, .2, .1]]], dtype=torch.float64)
    q_head, p_head = q[..., :2], p_log[..., :2].exp()
    q_tail, p_tail = 1 - q_head.sum(-1), 1 - p_head.sum(-1)
    coarsened_kl = (q_head * (q_head.log() - p_log[..., :2])).sum(-1)
    coarsened_kl = coarsened_kl + q_tail * (q_tail.log() - p_tail.log())
    coefficients = (q_head - (q_tail / p_tail)[..., None] * p_head).detach()
    correction = (coefficients * p_log[..., :2]).sum()
    kl_grad, = torch.autograd.grad(coarsened_kl.sum(), logits, retain_graph=True)
    correction_grad, = torch.autograd.grad(correction, logits)
    torch.testing.assert_close(kl_grad, -correction_grad)


@pytest.mark.parametrize('invalid', ['zero_current', 'zero_behavior', 'empty_row', 'nan', 'positive', 'beta_zero', 'beta_inf', 'beta_nan', 'shape', 'mask_dtype', 'reward_nan', 'integer'])
def test_bad_inputs_fail_explicitly(invalid):
    current, old = torch.full((2, 3), -1.), torch.full((2, 3), -2.)
    reward, mask, beta = torch.ones(2), torch.ones(2, 3, dtype=torch.bool), .1
    if invalid == 'zero_current': current[0, 0] = 0
    if invalid == 'zero_behavior': old[0, 0] = 0
    if invalid == 'empty_row': mask[0] = False
    if invalid == 'nan': current[0, 0] = float('nan')
    if invalid == 'positive': old[0, 0] = .1
    if invalid == 'beta_zero': beta = 0
    if invalid == 'beta_inf': beta = math.inf
    if invalid == 'beta_nan': beta = math.nan
    if invalid == 'shape': old = old[:, :2]
    if invalid == 'mask_dtype': mask = mask.float()
    if invalid == 'reward_nan': reward[0] = math.nan
    if invalid == 'integer': current = current.long()
    with pytest.raises(ValueError):
        klpo_sequence_loss(current, old, reward, mask, beta=beta)


def test_overflow_fails_instead_of_clipping_or_returning_nan():
    current = torch.tensor([[-1e-40]])
    with pytest.raises(FloatingPointError):
        klpo_sequence_loss(current, torch.tensor([[-1.]]), torch.ones(1), torch.ones(1, 1, dtype=torch.bool))
