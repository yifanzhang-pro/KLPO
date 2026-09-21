"""KLPO sequence/token regression with Binary KL, TopK-KL, MC-KL, or full KL.

All routes sum policy tokens and average complete trajectories, with no IS,
reward centering, ratio clipping, admission mask, or trajectory-length normalization.
Returned scalars are backward surrogates, not regression loss values.
"""

import math

import torch
from torch import Tensor

from ._validation import conditionals, finite_result, mc_records, prepare


def _log1mexp(x: Tensor) -> Tensor:
    """log(1-exp(x)) for x < 0, without subtracting a rounded probability."""
    return torch.where(x < -math.log(2), torch.log1p(-x.exp()), torch.log(-torch.expm1(x)))


@torch.no_grad()
def _binary_terms(current: Tensor, behavior: Tensor, action_mask: Tensor):
    if (current[action_mask] == 0).any() or (behavior[action_mask] == 0).any():
        raise ValueError("binary KL needs strictly negative logps; compute log_softmax in float32 or float64")
    # Safe non-boundary padding before log1mexp, division, or exponentiation.
    log_p = current.detach().masked_fill(~action_mask, -1.0)
    log_q = behavior.masked_fill(~action_mask, -1.0)
    log_pc, log_qc = _log1mexp(log_p), _log1mexp(log_q)
    ell = log_p - log_q
    q_complement = -torch.expm1(log_q)
    binary_kl = -log_q.exp() * ell + q_complement * (log_qc - log_pc)
    # Algebraically ell + k_bin; this form avoids cancellation near q=1.
    centered_ratio = q_complement * (ell + log_qc - log_pc)
    correction = (log_qc - log_pc).exp()
    return tuple(x.masked_fill(~action_mask, 0.0) for x in (binary_kl, centered_ratio, correction))


def klpo_sequence_loss(
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    action_mask: Tensor,
    *,
    beta: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """KLPO sequence regression with sampled-action Binary KL (optional in the report).

    log_probs/behavior_log_probs/action_mask: [B, T]; rewards: [B]. Each row
    contains a COMPLETE rollout. Only current log_probs receive gradients.
    Store the real collection logps, including sampling transforms, and keep
    them fixed during reuse. Recompute this function after every learner step.

    Binary KL requires strictly negative active logps (0 < p,q < 1). Score
    log-softmax in float32 or higher BEFORE gathering tokens: promoting an
    already rounded zero cannot recover its complement. No numerical clipping
    is applied. Masked slots may contain arbitrary NaN/inf values.
    """
    current, behavior, returns, lengths = prepare(
        log_probs, behavior_log_probs, rewards, action_mask, beta
    )
    binary_kl, centered_ratio, correction = _binary_terms(current, behavior, action_mask)
    with torch.no_grad():
        residual = returns - beta * centered_ratio.sum(-1)
        regression = residual.square() / (2 * beta)
    loss = -(residual * (correction * current).sum(-1)).mean()
    finite_result(loss, residual, correction, regression)
    return loss, {
        "residual": residual,
        "regression_loss": regression.mean(),
        "sequence_kl": binary_kl.sum(-1),
        "correction": correction,
        "policy_tokens": lengths,
    }


def klpo_sequence_full_loss(
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    action_mask: Tensor,
    *,
    full_log_probs: Tensor,
    behavior_full_log_probs: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Exact KLPO sequence regression with [B,T,V] full conditionals at stored histories.

    Both current log-probability tensors must share the model's autograd graph;
    sampled logps must be gathered from the supplied full conditionals. The KL
    derivative is retained. Population equivalence to canonical PMD assumes
    deterministic transitions, terminal rewards, and matched sampler rollouts.
    """
    current, behavior, returns, lengths = prepare(
        log_probs, behavior_log_probs, rewards, action_mask, beta
    )
    p_log, q_log = conditionals(
        full_log_probs, behavior_full_log_probs, action_mask, current.dtype, full=True
    )
    q = q_log.exp()
    # Convention 0 log 0 = 0; -inf may encode zero sampler support.
    safe_q_log = q_log.masked_fill(torch.isneginf(q_log), 0.0)
    local_kl = (q * (safe_q_log - p_log)).sum(-1).masked_fill(~action_mask, 0.0)
    residual = (returns - beta * (current - behavior + local_kl).sum(-1)).detach()
    regression = residual.square() / (2 * beta)
    loss = -(residual * (current + local_kl).sum(-1)).mean()
    finite_result(loss, residual, regression)
    return loss, {
        "residual": residual,
        "regression_loss": regression.mean(),
        "sequence_kl": local_kl.detach().sum(-1),
        "policy_tokens": lengths,
    }


def klpo_sequence_topk_loss(
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    action_mask: Tensor,
    *,
    conditional_log_probs: Tensor,
    behavior_conditional_log_probs: Tensor,
    full_vocabulary: bool = False,
    beta: float = 0.1,
    tail_floor: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """KLPO sequence regression with a Top-K Aggregated KL (TopK-KL).

    Uses D_K = R - beta * sum(ell + k_K) and differentiates the same k_K
    in -sg(D_K) * sum(log(p_action) + k_K). This is the sample gradient
    of D_K**2/(2*beta), including the tail stabilization when it is active.
    Conditional records are [B,T,K], aligned to the stored sampler head IDs;
    the sampled action need not belong to that head. Never renormalize it.

    For finite K, both tail masses are clamped to tail_floor in the KL scalar;
    autograd differentiates that literal stabilized KL (zero derivative through
    a clamped trainer tail). This differs from token regression's floored-ratio backward
    approximation at the floor. floored_tokens reports affected policy tokens.
    With full_vocabulary=True use exact full KL without a tail or floor.
    """
    if full_vocabulary:
        return klpo_sequence_full_loss(
            log_probs, behavior_log_probs, rewards, action_mask,
            full_log_probs=conditional_log_probs,
            behavior_full_log_probs=behavior_conditional_log_probs, beta=beta,
        )
    current, behavior, returns, lengths = prepare(
        log_probs, behavior_log_probs, rewards, action_mask, beta
    )
    if not math.isfinite(tail_floor) or not 0 < tail_floor < 1:
        raise ValueError("tail_floor must be finite and in (0, 1)")
    p_log, q_log = conditionals(
        conditional_log_probs, behavior_conditional_log_probs, action_mask,
        current.dtype, full=False,
    )
    p, q = p_log.exp(), q_log.exp()
    sampler_tail, trainer_tail = 1 - q.sum(-1), 1 - p.sum(-1)
    q_tail, p_tail = sampler_tail.clamp_min(tail_floor), trainer_tail.clamp_min(tail_floor)
    safe_q_log = q_log.masked_fill(torch.isneginf(q_log), 0.0)
    local_kl = (q * (safe_q_log - p_log)).sum(-1) + q_tail * (q_tail.log() - p_tail.log())
    local_kl = local_kl.masked_fill(~action_mask, 0.0)
    residual = (returns - beta * (current - behavior + local_kl).sum(-1)).detach()
    regression = residual.square() / (2 * beta)
    loss = -(residual * (current + local_kl).sum(-1)).mean()
    finite_result(loss, local_kl, residual, regression)
    return loss, {
        "residual": residual,
        "regression_loss": regression.mean(),
        "sequence_kl": local_kl.detach().sum(-1),
        "sampler_tail_mass": sampler_tail.detach().masked_fill(~action_mask, 0.0),
        "trainer_tail_mass": trainer_tail.detach().masked_fill(~action_mask, 0.0),
        "floored_tokens": (((sampler_tail < tail_floor) | (trainer_tail < tail_floor)) & action_mask).sum(),
        "policy_tokens": lengths,
    }


def klpo_sequence_mc_loss(
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    action_mask: Tensor,
    *,
    mc_log_probs: Tensor,
    behavior_mc_log_probs: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sequence regression with Monte Carlo KL and an unbiased cross estimate.

    MC records are [B,T,M], M >= 2, gathered at M IID draws WITH replacement
    from the recorded sampler at each visited prefix. Draw independently of
    the complete rollout, across positions and across columns j. Store IDs and
    original sampler logps; duplicate IDs are valid and must not be removed.
    Both trainer tensors must retain the same model graph. This function
    cannot infer whether sampling was independent from the supplied logps.

    For each column j form D_j = R-beta*sum_u(ell_u + log(q(v_uj)/p(v_uj))).
    Weight that column's corrected score by the mean D over OTHER columns.
    This removes covariance bias from reusing an MC estimate inside both
    a residual and its derivative. Averaging over auxiliary draws gives the
    full-KL sample gradient at a FIXED trainer and trajectory. Canonical PMD
    equivalence additionally needs the full-KL route assumptions.

    MC KL and the unbiased U-statistic regression_loss may be negative; do not
    clamp them. residual is the all-M mean, not each leave-one-out coefficient.
    Reusing fixed MC records gives a fixed empirical surrogate; resample from
    the historical sampler for a fresh conditional-unbiased gradient after an
    adaptive trainer update. No trajectory-length normalization is applied.
    """
    current, behavior, returns, lengths = prepare(
        log_probs, behavior_log_probs, rewards, action_mask, beta
    )
    p_log, q_log = mc_records(mc_log_probs, behavior_mc_log_probs, action_mask, current.dtype)
    m = p_log.shape[-1]
    if m < 2:
        raise ValueError("sequence MC-KL needs M >= 2 for leave-one-out residuals")
    with torch.no_grad():
        ratio = q_log - p_log.detach()
        residuals = returns[:, None] - beta * (
            (current.detach() - behavior).sum(-1)[:, None] + ratio.sum(1)
        )
        residual = residuals.mean(-1)
        centered = residuals - residual[:, None]
        other_residual = residual[:, None] - centered / (m - 1)
        regression = (residual.square() - centered.square().mean(-1) / (m - 1)) / (2 * beta)
    corrected = current.sum(-1)[:, None] - p_log.sum(1)
    loss = -(other_residual * corrected).mean(-1).mean()
    finite_result(loss, ratio, residuals, other_residual, regression)
    return loss, {
        "residual": residual,
        "regression_loss": regression.mean(),
        "sequence_kl": ratio.mean(-1).sum(-1),
        "mc_samples": torch.tensor(m, device=current.device),
        "policy_tokens": lengths,
    }


def klpo_token_loss(
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    action_mask: Tensor,
    *,
    kl_estimator: str = "mc",
    conditional_log_probs: Tensor | None = None,
    behavior_conditional_log_probs: Tensor | None = None,
    mc_log_probs: Tensor | None = None,
    behavior_mc_log_probs: Tensor | None = None,
    full_vocabulary: bool = False,
    beta: float = 0.1,
    tail_floor: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """KLPO token regression with Binary KL, TopK-KL, MC-KL, or full KL.

    kl_estimator='mc' (default) uses [B,T,M] MC records with M >= 1. Draw IID tokens
    with replacement from the historical sampler independently of the rollout,
    as in klpo_sequence_mc_loss. The correction is mean(log(p(v_j))), with
    uniform 1/M weights: do not multiply by q again or merge duplicate IDs.
    Independent auxiliary draws give the exact full-KL token gradient in
    expectation for a fixed trainer. Fixed records can be reused as an empirical
    surrogate; resample for conditional unbiasedness after adaptive updates.

    kl_estimator='binary' needs only sampled-action records and returns
    -mean(sum(sg(R-beta*ell) * sg((1-q)/(1-p)) * log(p))). The feedback
    remains per-token, not sequence regression's trajectory residual. It uses the same stable
    binary correction and strictly negative logp contract as klpo_sequence_loss.
    Binary correction is generally not an exact sampler-conditioned score
    mean; no full-KL population-equivalence claim applies to this approximation.

    kl_estimator='topk' or 'full' requires conditional records below.
    full_vocabulary=True remains supported as the top-K full-vocabulary limit.

    This is the report's default Steps 7--9: h = R - beta * ell per
    token, not the sequence-regression trajectory residual. Score centering supplies
    the correction shared by both routes; TopK-KL is a separate approximation. It uses the KL over
    K individual head actions plus one aggregate tail category:
    k_K = sum_H q_v log(q_v/p_v) + q_tail log(q_tail/p_tail).
    With unfloored tails, grad(k_K) = -grad(sum_H sg(q_v-rho*p_v)*log(p_v)).

    Gather current conditional logps at the STORED sampler head IDs. Preserve
    original head probabilities, without renormalization or inserting the
    sampled action. The action may lie outside the head. K=128 is the report's
    head size when TopK-KL is selected. Set full_vocabulary=True when K=V to use the exact
    sum and avoid a 0/0 tail ratio. Full KL and independent MC-KL have the exact
    population identity; finite K and the tail floor are approximations.
    """
    current, behavior, returns, lengths = prepare(
        log_probs, behavior_log_probs, rewards, action_mask, beta
    )
    if kl_estimator not in {"binary", "topk", "mc", "full"}:
        raise ValueError("kl_estimator must be binary, topk, mc, or full")
    with torch.no_grad():
        h = (returns[:, None] - beta * (current.detach() - behavior)).masked_fill(~action_mask, 0.0)
    if kl_estimator == "mc":
        if full_vocabulary or conditional_log_probs is not None or behavior_conditional_log_probs is not None:
            raise ValueError("MC-KL uses MC records, not head/full-vocabulary records")
        if mc_log_probs is None or behavior_mc_log_probs is None:
            raise ValueError("MC-KL requires current and historical MC records")
        p_log, q_log = mc_records(mc_log_probs, behavior_mc_log_probs, action_mask, current.dtype)
        local_kl = (q_log - p_log.detach()).mean(-1)
        loss = -(h * (current - p_log.mean(-1))).sum(-1).mean()
        finite_result(loss, h, local_kl)
        return loss, {
            "return_coefficient": h,
            "sequence_kl": local_kl.sum(-1),
            "mc_samples": torch.tensor(p_log.shape[-1], device=current.device),
            "policy_tokens": lengths,
        }
    if mc_log_probs is not None or behavior_mc_log_probs is not None:
        raise ValueError("MC records require kl_estimator='mc'")
    if kl_estimator == "binary":
        if full_vocabulary or conditional_log_probs is not None or behavior_conditional_log_probs is not None:
            raise ValueError("binary KL uses sampled-action records only; omit conditional records/full_vocabulary")
        binary_kl, _, correction = _binary_terms(current, behavior, action_mask)
        loss = -(h * correction * current).sum(-1).mean()
        finite_result(loss, h, correction)
        return loss, {
            "return_coefficient": h,
            "sequence_kl": binary_kl.sum(-1),
            "correction": correction,
            "policy_tokens": lengths,
        }
    if conditional_log_probs is None or behavior_conditional_log_probs is None:
        raise ValueError("topk/full KL requires current and historical conditional records")
    full_vocabulary = full_vocabulary or kl_estimator == "full"
    if not math.isfinite(tail_floor) or not 0 < tail_floor < 1:
        raise ValueError("tail_floor must be finite and in (0, 1)")
    p_log, q_log = conditionals(
        conditional_log_probs, behavior_conditional_log_probs, action_mask,
        current.dtype, full=full_vocabulary,
    )
    with torch.no_grad():
        q, p = q_log.exp(), p_log.detach().exp()
        safe_q_log = q_log.masked_fill(torch.isneginf(q_log), 0.0)
        local_kl = (q * (safe_q_log - p_log.detach())).sum(-1)
        sampler_tail = (1 - q.sum(-1)).masked_fill(~action_mask, 0.0)
        trainer_tail = (1 - p.sum(-1)).masked_fill(~action_mask, 0.0)
        if full_vocabulary:
            coefficients = q
            floored = torch.zeros_like(action_mask)
        else:
            floored = ((sampler_tail < tail_floor) | (trainer_tail < tail_floor)) & action_mask
            rho = sampler_tail.clamp_min(tail_floor) / trainer_tail.clamp_min(tail_floor)
            coefficients = q - rho[..., None] * p
            q_tail, p_tail = sampler_tail.clamp_min(tail_floor), trainer_tail.clamp_min(tail_floor)
            local_kl += q_tail * (q_tail.log() - p_tail.log())
        local_kl = local_kl.masked_fill(~action_mask, 0.0)
        coefficients = coefficients.masked_fill(~action_mask[..., None], 0.0)
    correction = (coefficients * p_log).sum(-1)
    loss = -(h * (current - correction)).sum(-1).mean()
    finite_result(loss, h, coefficients)
    return loss, {
        "return_coefficient": h,
        "sequence_kl": local_kl.sum(-1),
        "sampler_tail_mass": sampler_tail,
        "trainer_tail_mass": trainer_tail,
        "floored_tokens": floored.sum(),
        "policy_tokens": lengths,
    }
