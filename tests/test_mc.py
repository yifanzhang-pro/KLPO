"""Finite enumeration of auxiliary draws, independent of implementation formulas."""

from itertools import product

import pytest
import torch

from klpo import klpo_sequence_full_loss, klpo_sequence_mc_loss, klpo_token_loss


@pytest.mark.parametrize('route,m', [('sequence', 2), ('sequence', 3), ('token', 1), ('token', 2)])
@pytest.mark.parametrize('steps', [1, 2])
def test_mc_expected_gradient_matches_full_kl_with_shared_parameters(route, m, steps):
    theta = torch.tensor([.5, -.7, .2], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[[1., -.4, .2], [-.2, .3, 1.]],
                            [[.3, 1., -.5], [.7, -.3, .4]]], dtype=theta.dtype)[:steps]
    p = (weights @ theta).log_softmax(-1)[None]
    q = torch.tensor([[.25, .75], [.6, .4]], dtype=theta.dtype)[:steps][None]
    actions = torch.tensor([[1, 0]])[:, :steps]
    mask = torch.ones(1, steps, dtype=torch.bool)
    beta = .6
    args = (p.gather(-1, actions[..., None]).squeeze(-1),
            q.log().gather(-1, actions[..., None]).squeeze(-1), torch.tensor([.35]), mask)
    if route == 'sequence':
        exact, stats = klpo_sequence_full_loss(*args, full_log_probs=p,
                                              behavior_full_log_probs=q.log(), beta=beta)
    else:
        exact, stats = klpo_token_loss(*args, conditional_log_probs=p,
            behavior_conditional_log_probs=q.log(), kl_estimator='full', beta=beta)
    expected = 0.
    expected_regression = 0.
    expected_kl = 0.
    naive = 0.
    for values in product(range(2), repeat=steps * m):
        ids = torch.tensor(values).reshape(1, steps, m)
        old = q.log().gather(-1, ids)
        probability = old.exp().prod()
        records = dict(mc_log_probs=p.gather(-1, ids), behavior_mc_log_probs=old)
        if route == 'sequence':
            loss, mc = klpo_sequence_mc_loss(*args, **records, beta=beta)
            expected_regression += probability * mc['regression_loss']
            residual = args[2] - beta * (args[0] - args[1] + (old - records['mc_log_probs']).mean(-1)).sum(-1)
            naive += probability * residual.square().mean() / (2 * beta)
        else:
            loss, mc = klpo_token_loss(*args, **records, beta=beta)
        expected += probability * loss
        expected_kl += probability * mc['sequence_kl']
    actual_grad, = torch.autograd.grad(expected, theta, retain_graph=True)
    exact_grad, = torch.autograd.grad(exact, theta, retain_graph=True)
    torch.testing.assert_close(actual_grad, exact_grad, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(expected_kl, (q * (q.log() - p.detach())).sum((-1, -2)))
    if route == 'sequence':
        torch.testing.assert_close(expected_regression, stats['regression_loss'], atol=1e-12, rtol=1e-11)
        naive_grad, = torch.autograd.grad(naive, theta)
        assert (naive_grad - exact_grad).norm() > 1e-4


@pytest.mark.parametrize('m', [2, 5])
def test_sequence_surrogate_matches_pairwise_u_statistic_and_masks_padding(m):
    torch.manual_seed(13)
    logits = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    p = logits.log_softmax(-1)
    q = torch.tensor([.7, .2, .1], dtype=torch.float64).log().expand_as(p).clone().requires_grad_()
    mask = torch.tensor([[True, False, False], [True, True, True]])
    ids = torch.randint(0, 3, (2, 3, m))
    pl, ql = p.gather(-1, ids), q.gather(-1, ids)
    current = p[..., 0].masked_fill(~mask, float('nan'))
    old = q[..., 0].masked_fill(~mask, float('inf'))
    rewards = torch.tensor([.4, -.7], dtype=p.dtype, requires_grad=True)
    beta = .4
    loss, stats = klpo_sequence_mc_loss(current, old, rewards, mask,
        mc_log_probs=pl.masked_fill(~mask[..., None], float('nan')),
        behavior_mc_log_probs=ql.masked_fill(~mask[..., None], float('-inf')), beta=beta)
    d = rewards.detach()[:, None] - beta * (
        ((p[..., 0] - q.detach()[..., 0]) * mask).sum(-1)[:, None]
        + ((ql.detach() - pl) * mask[..., None]).sum(1))
    pairs = torch.stack([d[:, j] * d[:, k] for j in range(m) for k in range(m) if j != k])
    objective = pairs.mean() / (2 * beta)
    expected, = torch.autograd.grad(objective, logits, retain_graph=True)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(stats['regression_loss'], objective.detach())
    assert (logits.grad[~mask] == 0).all()
    assert q.grad is None and rewards.grad is None
    assert all(not value.requires_grad for value in stats.values())


def test_negative_mc_kl_and_u_statistic_are_not_clamped_and_duplicates_are_kept():
    logits = torch.tensor([[[.8, .2]]], dtype=torch.float64).log().requires_grad_()
    p = logits.log_softmax(-1)
    q = torch.tensor([[[.25, .75]]], dtype=p.dtype).log()
    mask = torch.ones(1, 1, dtype=torch.bool)
    ids = torch.tensor([[[0, 0, 0, 0]]])
    loss, stats = klpo_token_loss(p[..., 1], q[..., 1], torch.ones(1), mask,
        kl_estimator='mc', mc_log_probs=p.gather(-1, ids), behavior_mc_log_probs=q.gather(-1, ids))
    assert stats['sequence_kl'].item() < 0
    grad, = torch.autograd.grad(loss, logits, retain_graph=True)
    assert grad.norm() > 0
    # Set the mean residual to zero; its cross-product estimate is negative.
    reward = (.3 * (p[..., 1] - q[..., 1] + (q - p).mean(-1)).sum(-1)).detach()
    _, stats = klpo_sequence_mc_loss(p[..., 1], q[..., 1], reward, mask,
        mc_log_probs=p, behavior_mc_log_probs=q, beta=.3)
    assert stats['regression_loss'].item() < 0


@pytest.mark.parametrize('route', ['sequence', 'token'])
@pytest.mark.parametrize('bad', ['shape', 'empty', 'integer', 'nan', 'posinf', 'neginf', 'positive', 'mismatch'])
def test_invalid_mc_records(route, bad):
    args = (torch.full((1, 2), -1.), torch.full((1, 2), -2.),
            torch.ones(1), torch.ones(1, 2, dtype=torch.bool))
    current = torch.full((1, 2, 2), -1.)
    old = current.clone()
    if bad == 'shape': current = current[0]
    elif bad == 'empty': current, old = current[..., :0], old[..., :0]
    elif bad == 'integer': current = current.long()
    elif bad == 'mismatch': old = old[..., :1]
    else: old[0, 0, 0] = {'nan': float('nan'), 'posinf': float('inf'), 'neginf': -float('inf'), 'positive': .1}[bad]
    with pytest.raises(ValueError):
        if route == 'sequence':
            klpo_sequence_mc_loss(*args, mc_log_probs=current, behavior_mc_log_probs=old)
        else:
            klpo_token_loss(*args, kl_estimator='mc', mc_log_probs=current, behavior_mc_log_probs=old)


def test_singleton_sequence_and_mixed_record_contracts_rejected():
    args = (torch.full((1, 1), -1.), torch.full((1, 1), -2.),
            torch.ones(1), torch.ones(1, 1, dtype=torch.bool))
    rec = torch.full((1, 1, 1), -1.)
    with pytest.raises(ValueError, match='M >= 2'):
        klpo_sequence_mc_loss(*args, mc_log_probs=rec, behavior_mc_log_probs=rec)
    for kwargs in [dict(kl_estimator='mc'), dict(kl_estimator='mc', mc_log_probs=rec),
                   dict(kl_estimator='mc', mc_log_probs=rec, behavior_mc_log_probs=rec, full_vocabulary=True),
                   dict(kl_estimator='mc', mc_log_probs=rec, behavior_mc_log_probs=rec, conditional_log_probs=rec),
                   dict(kl_estimator='binary', mc_log_probs=rec), dict(kl_estimator='tk')]:
        with pytest.raises(ValueError):
            klpo_token_loss(*args, **kwargs)


def test_no_legacy_estimator_api_alias():
    import klpo
    import klpo.loss
    assert not hasattr(klpo, 'klpo_sequence_tk_loss')
    assert not hasattr(klpo.loss, 'klpo_sequence_tk_loss')
