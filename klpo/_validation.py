"""Shared tensor contracts; mask padding before any nonlinear arithmetic."""

import math

import torch
from torch import Tensor


def prepare(log_probs, behavior_log_probs, rewards, action_mask, beta):
    if log_probs.ndim != 2 or not log_probs.numel():
        raise ValueError("log_probs must be nonempty [B, T]")
    if behavior_log_probs.shape != log_probs.shape or action_mask.shape != log_probs.shape:
        raise ValueError("log probabilities and action_mask must share [B, T]")
    if rewards.shape != (log_probs.shape[0],):
        raise ValueError("rewards must have shape [B]")
    if action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be boolean")
    if any(t.device != log_probs.device for t in (behavior_log_probs, rewards, action_mask)):
        raise ValueError("all inputs must be on the same device")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    lengths = action_mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("every row must contain a complete trajectory with policy tokens")
    for values in (log_probs, behavior_log_probs):
        if not values.is_floating_point():
            raise ValueError("log probabilities must be floating point")
        valid = values[action_mask]
        if not torch.isfinite(valid).all() or (valid > 0).any():
            raise ValueError("sampled-action log probabilities must be finite and <= 0")
    dtype = torch.promote_types(log_probs.dtype, behavior_log_probs.dtype)
    dtype = torch.promote_types(dtype, torch.float32)
    current = torch.where(action_mask, log_probs.to(dtype), 0.0)
    behavior = torch.where(action_mask, behavior_log_probs.detach().to(dtype), 0.0)
    return current, behavior, rewards.detach().to(dtype), lengths


def conditionals(current: Tensor, behavior: Tensor, mask: Tensor, dtype, *, full: bool):
    if current.ndim != 3 or current.shape[:2] != mask.shape or current.shape[-1] < 1:
        raise ValueError("conditional log probabilities must have shape [B, T, K]")
    if behavior.shape != current.shape:
        raise ValueError("trainer and sampler conditional records must have equal shapes")
    for values in (current, behavior):
        if values.device != mask.device or not values.is_floating_point():
            raise ValueError("conditional records must be floating point on the input device")
        valid = values[mask]
        # Zero sampler probabilities (-inf) are permitted; trainer support must be positive.
        if torch.isnan(valid).any() or (valid > 0).any():
            raise ValueError("conditional log probabilities must be <= 0 and not NaN")
    if not torch.isfinite(current[mask]).all():
        raise ValueError("trainer conditionals must have finite log probabilities")
    p_log = torch.where(mask[..., None], current.to(dtype), 0.0)
    q_log = torch.where(mask[..., None], behavior.detach().to(dtype), 0.0)
    with torch.no_grad():
        for values in (p_log, q_log):
            mass = values[mask].exp().sum(-1)
            if full:
                if not torch.allclose(mass, torch.ones_like(mass), rtol=1e-5, atol=1e-6):
                    raise ValueError("full conditional records must sum to one")
            elif (mass > 1 + 1e-6).any():
                raise ValueError("top-K head mass must not exceed one; use original probabilities")
    return p_log, q_log


def finite_result(loss, *values):
    if not torch.isfinite(loss).all() or any(not torch.isfinite(x).all() for x in values):
        raise FloatingPointError("KLPO arithmetic overflow; inspect probabilities/rewards or use float64")


def mc_records(current: Tensor, behavior: Tensor, mask: Tensor, dtype):
    """Validate IID records with replacement; independence is a collection contract."""
    if current.ndim != 3 or current.shape[:2] != mask.shape or current.shape[-1] < 1:
        raise ValueError("MC log probabilities must have shape [B, T, M] with M >= 1")
    if behavior.shape != current.shape:
        raise ValueError("trainer and sampler MC records must have equal shapes")
    for values in (current, behavior):
        if values.device != mask.device or not values.is_floating_point():
            raise ValueError("MC records must be floating point on the input device")
        valid = values[mask]
        if not torch.isfinite(valid).all() or (valid > 0).any():
            raise ValueError("active MC log probabilities must be finite and <= 0")
        dtype = torch.promote_types(dtype, values.dtype)
    # Repeated IDs are valid. Summing their probabilities can exceed one.
    return (torch.where(mask[..., None], current.to(dtype), 0.0),
            torch.where(mask[..., None], behavior.detach().to(dtype), 0.0))
