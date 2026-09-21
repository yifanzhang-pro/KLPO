"""Optional Step 10 primitives for a COMPLETE optimizer displacement.

These functions neither construct an optimizer proposal nor run a model JVP.
The caller supplies J_logits @ D, including LR, momentum, preconditioning,
and weight decay. They do not implement a historical-sampler trust gate.
"""

import math

import torch
from torch import Tensor


@torch.no_grad()
def predicted_kl(logits: Tensor, logit_jvp: Tensor, probe_mask: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Mean quadratic current-to-proposed KL on fixed probe histories.

    logits and logit_jvp: [..., V]; probe_mask: boolean [...]. For distributed
    probes, aggregate variance sums and token counts globally before scaling.
    This helper is local; it never performs an implicit distributed reduction.
    """
    if logits.ndim < 2 or not logits.numel() or logits.shape != logit_jvp.shape:
        raise ValueError('logits and logit_jvp must share nonempty [..., V] shape')
    if probe_mask.shape != logits.shape[:-1] or probe_mask.dtype != torch.bool:
        raise ValueError('probe_mask must be boolean with shape [...]')
    if any(x.device != logits.device for x in (logit_jvp, probe_mask)):
        raise ValueError('probe tensors must be on the same device')
    if not logits.is_floating_point() or not logit_jvp.is_floating_point():
        raise ValueError('probe logits and JVP must be floating point')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    dtype = torch.promote_types(torch.promote_types(logits.dtype, logit_jvp.dtype), torch.float32)
    if not probe_mask.any():
        return torch.full((), float('nan'), dtype=dtype, device=logits.device)
    z, dz = logits[probe_mask].to(dtype), logit_jvp[probe_mask].to(dtype)
    p = (z / temperature).softmax(-1)
    mean = (p * dz).sum(-1, keepdim=True)
    return (p * (dz - mean).square()).sum(-1).mean() / (2 * temperature ** 2)


def kl_budget_scale(prediction: Tensor | float | None, budget: float | None, *, alpha_max: float = 1.0) -> float:
    """Return alpha; None budget disables calibration, invalid probes skip.

    A zero/NaN/inf/unavailable/negative prediction returns 0, never a full step.
    The caller should flag an invalid probe and record realized KL separately.
    """
    if budget is None:
        return 1.0
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('budget must be nonnegative and finite, or None to disable')
    if not math.isfinite(alpha_max) or not 0 < alpha_max <= 1:
        raise ValueError('alpha_max must be in (0, 1]')
    q = float(prediction) if prediction is not None else float('nan')
    if budget == 0 or not math.isfinite(q) or q <= 0:
        return 0.0
    return min(alpha_max, math.sqrt(budget / q))
