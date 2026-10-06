"""Enumerate a deterministic variable-length tree to check the paper identities."""

import torch

from klpo import klpo_sequence_full_loss, klpo_sequence_loss, klpo_token_loss


def test_exact_sequence_token_and_profiled_pmd_population_gradients():
    # Root action 0 terminates, action 1 continues to a second binary decision.
    q = torch.tensor([[.3, .7], [.6, .4]], dtype=torch.float64)
    returns = torch.tensor([.2, 1.1, -.5], dtype=torch.float64)
    trajectory_prob = torch.stack((q[0, 0], q[0, 1] * q[1, 0], q[0, 1] * q[1, 1]))
    logits = torch.tensor([[.5, -.2], [-.6, .9]], dtype=torch.float64, requires_grad=True)
    p_log, q_log, beta = logits.log_softmax(-1), q.log(), .4
    mask = torch.tensor([[True, False], [True, True], [True, True]])
    actions = torch.tensor([[0, 0], [1, 0], [1, 1]])
    full = p_log[None].expand(3, -1, -1)
    old_full = q_log[None].expand_as(full)
    sampled = full.gather(-1, actions[..., None]).squeeze(-1)
    sampled_old = old_full.gather(-1, actions[..., None]).squeeze(-1)
    full_terms, sc_terms, binary_terms = [], [], []
    for i in range(3):
        row = slice(i, i + 1)
        args = (sampled[row], sampled_old[row], returns[row], mask[row])
        full_loss, _ = klpo_sequence_full_loss(*args, full_log_probs=full[row], behavior_full_log_probs=old_full[row], beta=beta)
        sc_loss, _ = klpo_token_loss(*args, kl_estimator='full', conditional_log_probs=full[row],
            behavior_conditional_log_probs=old_full[row], full_vocabulary=True, beta=beta)
        binary_loss, _ = klpo_sequence_loss(*args, beta=beta)
        full_terms.append(full_loss)
        sc_terms.append(sc_loss)
        binary_terms.append(binary_loss)
    v_child = q[1] @ returns[1:]
    root_q = torch.stack((returns[0], v_child))
    a_root = root_q - q[0] @ root_q
    a_child = returns[1:] - v_child
    advantage = torch.stack((a_root, a_child))
    ell = p_log - q_log
    kl = (q * -ell).sum(-1)
    delta = advantage / beta - ell - kl[:, None]
    visitation = torch.stack((torch.tensor(1.), q[0, 1]))
    canonical = beta / 2 * ((q * delta.square()).sum(-1) * visitation).sum()
    expected, = torch.autograd.grad(canonical, logits, retain_graph=True)
    # With a binary vocabulary, sampled-action binary KL is the full local KL.
    for terms in (full_terms, sc_terms, binary_terms):
        estimate = (trajectory_prob * torch.stack(terms)).sum()
        actual, = torch.autograd.grad(estimate, logits, retain_graph=True)
        torch.testing.assert_close(actual, expected)
    # Verify the omitted value changes only the scalar constant.
    local_kl = (q[None] * (old_full - full)).sum(-1)
    d = returns - beta * ((sampled - sampled_old + local_kl) * mask).sum(-1)
    trajectory_loss = (trajectory_prob * d.square()).sum() / (2 * beta)
    value = q[0] @ root_q
    torch.testing.assert_close(trajectory_loss, canonical + value.square() / (2 * beta))
