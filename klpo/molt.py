"""KLPO loss implementation for the labs-molt fork's native KL record interface.

The backend calls this optional adapter for both routes and all four estimators. It
does not import Molt, Ray, transformers, or CUDA, so its gradients can be checked
on CPU. Raw terminal rewards are passed separately from Molt's advantages.
"""

import math

import torch
from torch import Tensor, nn

from .loss import (klpo_sequence_loss, klpo_sequence_topk_loss, klpo_sequence_full_loss,
                   klpo_sequence_mc_loss, klpo_token_loss)


class KLPOLoss(nn.Module):
    loss_agg_mode = "seq-mean-token-sum"

    def __init__(self, beta: float = 0.1, *, route: str = "token",
                 kl_estimator: str = "mc", tail_floor: float = 1e-6):
        super().__init__()
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        if route not in {"sequence", "token"}:
            raise ValueError("route must be sequence or token")
        if kl_estimator not in {"binary", "topk", "mc", "full"}:
            raise ValueError("kl_estimator must be binary, topk, mc, or full")
        if not 0 < tail_floor < 1:
            raise ValueError("tail_floor must be in (0, 1)")
        self.kl_estimator = kl_estimator
        self.tail_floor = tail_floor
        self.beta = beta
        self.route = route

    def forward(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor | None,
        advantages: Tensor | None,
        action_mask: Tensor | None = None,
        rollout_log_probs: Tensor | None = None,
        dp_size: int = 1,
        batch_num_tokens: Tensor | None = None,
        global_batch_size: Tensor | int | None = None,
        *,
        rewards: Tensor,
        kl_log_probs: Tensor | None = None,
        behavior_kl_log_probs: Tensor | None = None,
        full_vocabulary: bool = False,
    ):
        if rollout_log_probs is None or action_mask is None:
            raise ValueError("KLPO requires actual rollout logps and a complete-trajectory action mask")
        if global_batch_size is None:
            raise ValueError("global_batch_size must count the entire optimizer-step batch across DP ranks")
        if not isinstance(dp_size, int) or dp_size < 1:
            raise ValueError("dp_size must be a positive integer")
        count = torch.as_tensor(global_batch_size, device=log_probs.device).detach()
        if count.ndim or not torch.isfinite(count) or count < log_probs.shape[0] or count != count.round():
            raise ValueError("global_batch_size must be a positive integer >= the local trajectory count")
        args = (log_probs, rollout_log_probs, rewards, action_mask.bool())
        options = {"beta": self.beta}
        if self.kl_estimator == "mc":
            options.update(mc_log_probs=kl_log_probs, behavior_mc_log_probs=behavior_kl_log_probs)
        elif self.kl_estimator in {"topk", "full"}:
            if self.route == "sequence" and self.kl_estimator == "full":
                options.update(full_log_probs=kl_log_probs, behavior_full_log_probs=behavior_kl_log_probs)
            else:
                options.update(conditional_log_probs=kl_log_probs,
                               behavior_conditional_log_probs=behavior_kl_log_probs,
                               full_vocabulary=full_vocabulary, tail_floor=self.tail_floor)
        elif kl_log_probs is not None or behavior_kl_log_probs is not None or full_vocabulary:
            raise ValueError("Binary KL does not consume auxiliary records")
        if self.route == "sequence":
            loss_fn = {"binary": klpo_sequence_loss, "topk": klpo_sequence_topk_loss,
                       "mc": klpo_sequence_mc_loss, "full": klpo_sequence_full_loss}[self.kl_estimator]
            loss, stats = loss_fn(*args, **options)
        else:
            loss, stats = klpo_token_loss(*args, kl_estimator=self.kl_estimator, **options)
        # Local mean -> local sum / global B, compensating DDP/FSDP averaging.
        # Molt must NOT divide this again by the gradient accumulation count.
        scaled_loss = loss * (log_probs.shape[0] * dp_size / count)
        zero = loss.detach().new_zeros(())
        conditional_kl = stats['sequence_kl'].sum() / stats['policy_tokens'].sum()
        # Token regression's backward surrogate is not a squared regression loss.
        reported = stats['regression_loss'] if self.route == "sequence" else loss.detach()
        return scaled_loss, reported, zero, conditional_kl, conditional_kl, zero
