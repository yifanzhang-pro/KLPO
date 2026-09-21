# KLPO algorithms and loss APIs

[Project home](../README.md) · [Paper](../KLPO.pdf) · [Training guide](training.md)

Default: **KLPO token regression + Monte Carlo KL (MC-KL)**. This reference documents the mathematical surrogates, all eight route/estimator combinations, and their tensor contracts.

## Default: KLPO token regression + MC-KL

This is the paper's Figure 1 and the default for the library, CPU example, and
Molt launcher. Let `p` be the current trainer policy, `q` the actual historical
sampler, `R` the terminal reward, and `ell = log(p_action) - log(q_action)`.
M counts auxiliary token draws per prefix, not full response rollouts; the
launchers default to M=128, and token regression allows any M >= 1.

At each visited prefix draw `v_j ~ q` IID **with replacement**, independently of
the complete rollout and across prefixes. Store the M token IDs and their
original sampler log-probabilities, then gather trainer logps at those IDs:

```text
k_MC = mean_j(log(q(v_j)) - log(p(v_j)))
```

This is an unbiased estimate of full local KL for any M >= 1, but an individual
estimate may be negative. Do not clamp it, renormalize the sampled probabilities,
multiply the summands by q, deduplicate IDs, or use the rollout action as a draw.
Repeated IDs are valid even when M exceeds the vocabulary size.

**Token regression** directly averages the sampled scores:

```text
loss_token_mc = -mean_responses(sum_tokens(stopgrad(R-beta*ell)
                               * (log(p_action) - mean_j(log(p(v_j))))))
```

It has the full-KL token gradient in expectation over independent MC draws, even
for M=1.

```python
from klpo import klpo_token_loss

# logps, sampler_logps, action_mask: [B,T]; rewards: [B].
# stored_mc_ids and stored_mc_logps: [B,T,M], independently sampled from q.
# Current trainer logps retain autograd; historical sampler records stay fixed.
loss, stats = klpo_token_loss(
    logps, sampler_logps, rewards, action_mask,
    mc_log_probs=current_full_logps.gather(-1, stored_mc_ids),
    behavior_mc_log_probs=stored_mc_logps,
    beta=0.1,  # kl_estimator="mc" is the default.
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Each row must be a **complete trajectory**. Sum all generated policy tokens,
including termination consistently; exclude prompt, padding, and tool-output
tokens with a boolean mask. Average over responses, **without dividing by length**.
Keep the collection probabilities and sampler version fixed during reuse.

There is no action importance multiplier, batch/group reward centering, policy
ratio clipping, critic, reference-policy pass, or sequence admission gate.

Fixed MC records can be reused as an empirical surrogate. After the trainer has
adapted to them, a fresh conditional-unbiased MC estimate requires fresh draws
from the **historical** sampler. The toy example resamples at every learner step
using its stored historical full conditionals. It does not switch the sampler
version when replaying a response. Neither route needs a critic or normalizer.

## Optional routes and KL estimators

**KLPO token regression** uses a separate return coefficient for each token.
**KLPO sequence regression** shares one trajectory residual across all tokens.
Score centering is the correction operation shared by both. **Monte Carlo KL (MC-KL)**,
**Top-K Aggregated KL (TopK-KL)**, **Binary KL**, and **Full KL**
specify the conditional-KL estimator independently;
all eight combinations are implemented in the library and CPU example.

**M is the number of IID random samples per prefix; K is the head size.**
MC-KL averages log-ratios of M independent sampler draws with replacement,
without a head, tail bucket, or probability reweighting. TopK-KL keeps the sampler's
K highest-probability tokens and aggregates all other tokens into one tail bucket.
MC tokens are sampled independently of the rollout action and continuation;
they are extra token draws, not extra response rollouts.

The report's Figure 1 defaults to **KLPO token regression + MC-KL**;
Figure 2 shows **KLPO sequence regression + MC-KL** as the alternative route.
TopK-KL and Binary KL are optional approximations for both. The toy CLI, Molt
launcher, native Molt CLI, and `KLPOLoss` adapter default to token regression
with MC-KL; CLI entry points use M=128, configurable with `--mc-samples` (native Molt:
`--actor.klpo_mc_samples`). `klpo_token_loss` also defaults to MC-KL and requires
MC records. The sampled-action-only `klpo_sequence_loss` API remains Binary KL;
use `klpo_sequence_mc_loss` for sequence MC-KL.

KLPO sequence regression uses a path sum of local KLs and targets local PMD.
It is distinct from the paper's SKLPO response-Gibbs objective, which requires
prompt-and-version normalization.

| Route | KL estimator | API | Feedback |
| --- | --- | --- | --- |
| KLPO token regression | MC-KL (default) | `klpo_token_loss` | Per-token `R - beta*ell`; M >= 1 |
| KLPO token regression | TopK-KL | `klpo_token_loss(..., kl_estimator="topk")` | Per-token `R - beta*ell` |
| KLPO token regression | Binary KL | `klpo_token_loss(..., kl_estimator="binary")` | Per-token `R - beta*ell` |
| KLPO token regression | Full KL | Same API, `kl_estimator="full"` | Per-token `R - beta*ell` |
| KLPO sequence regression | MC-KL | `klpo_sequence_mc_loss` | Leave-one-out trajectory residuals; M >= 2 |
| KLPO sequence regression | TopK-KL | `klpo_sequence_topk_loss` | Trajectory residual `D_K` |
| KLPO sequence regression | Binary KL | `klpo_sequence_loss` | Trajectory residual `D` |
| KLPO sequence regression | Full KL | `klpo_sequence_full_loss` | Trajectory residual `D` |

Choose another combination explicitly:

```bash
python examples/train_toy.py --kl-estimator topk --top-k 2
python examples/train_toy.py --kl-estimator binary
python examples/train_toy.py --kl-estimator full --kl-budget 1e-4
python examples/train_toy.py --route sequence --mc-samples 8
python examples/train_toy.py --route sequence --kl-estimator topk --top-k 2
python examples/train_toy.py --route sequence --kl-estimator binary
python examples/train_toy.py --route sequence --kl-estimator full --kl-budget 1e-4
```

For a deterministic local KL `k`, the routes differ in their detached feedback:

```text
Token regression:    h = R - beta * ell                         # per token
     gradient = -mean_responses(sum_tokens(h * (grad(log(p_action)) + grad(k))))
Sequence regression: D = R - beta * sum_tokens(ell + k)
     gradient = -mean_responses(D * sum_tokens(grad(log(p_action)) + grad(k)))
```

Token regression's TopK-KL implementation uses the floored-ratio approximation to `grad(k)`
described below. Do not replace its token feedback with sequence regression's trajectory residual.

### KLPO token regression + TopK-KL

TopK-KL retains the sampler's K highest-probability tokens and aggregates the
rest into one tail bucket. With unclamped positive tails:

```text
rho = q_tail/p_tail
C   = sum_head(stopgrad(q_v-rho*p_v) * log(p_v))
loss_token = -mean_responses(sum_tokens(stopgrad(R-beta*ell) * (log(p_action)-C)))
```

The derivative of `C` is the negative derivative of this coarsened KL when the
tail floor is inactive. Token regression uses this KL derivative to approximate the
sampler-conditioned mean score. KLPO sequence regression + TopK-KL uses a trajectory residual built
from `k_K`; KLPO token regression + TopK-KL keeps the local coefficient `R-beta*ell`.

### KLPO token regression + Binary KL

```text
k_bin = q_action log(q_action/p_action) + (1-q_action) log((1-q_action)/(1-p_action))
h = R - beta * ell
omega = (1-q_action)/(1-p_action)
loss_token_binary = -mean_responses(sum_tokens(stopgrad(h) * stopgrad(omega) * log(p_action)))
```

This follows `grad(log(p_action) + k_bin) = omega * grad(log(p_action))`.
It needs only sampled-action logps and uses the same stable complement arithmetic
as KLPO sequence regression + Binary KL. There is no trajectory residual in this coefficient:

```python
from klpo import klpo_token_loss

loss, stats = klpo_token_loss(
    logps, sampler_logps, rewards, action_mask, kl_estimator="binary", beta=0.1,
)
```

The binary correction generally fails to center the score exactly under the full
sampler distribution. It is an approximation, not an exact critic-free population
gradient. The returned scalar is a backward surrogate; no regression-loss value
is reported for this token-regression variant.

### KLPO sequence regression + MC-KL

This is the paper's Figure 2 alternative. **Sequence regression** must avoid multiplying correlated estimates:
plugging the same mean KL into a squared residual adds a trainer-dependent
variance term. We use an unbiased leave-one-out construction for M >= 2:

```text
D_j = R - beta * sum_tokens(ell + log(q(v_j)) - log(p(v_j)))
D_minus_j = mean_{l != j}(D_l)
loss_sequence_mc = -mean_responses(mean_j(stopgrad(D_minus_j)
                                         * sum_tokens(log(p_action) - log(p(v_j)))))
```

The M columns must be independent conditional on the complete trajectory. This
recovers the full-KL sequence gradient in expectation over auxiliary draws at a
fixed trainer. Together with the report's deterministic-transition and terminal-
reward assumptions, the two MC routes have the same population gradient.
`stats["regression_loss"]` is the unbiased U-statistic
`mean_{j != l}(D_j * D_l)/(2*beta)`, which can be negative; `stats["residual"]`
is the all-M mean. The returned scalar is a backward surrogate.

```python
from klpo import klpo_sequence_mc_loss

loss, stats = klpo_sequence_mc_loss(
    logps, sampler_logps, rewards, action_mask,
    mc_log_probs=current_full_logps.gather(-1, stored_mc_ids),
    behavior_mc_log_probs=stored_mc_logps,
    beta=0.1,  # M >= 2 for leave-one-out residuals.
)
```

### KLPO sequence regression + TopK-KL

TopK-KL partitions the vocabulary into K head actions plus an aggregate tail:

```text
k_K = sum_head(q_v log(q_v/p_v)) + q_tail log(q_tail/p_tail)
D_K = R - beta * sum_tokens(ell + k_K)
loss_sequence_topk = -mean_responses(stopgrad(D_K) * sum_tokens(log(p_action) + k_K))
```

Recompute `D_K` at each learner step. Differentiate through both current
log-probability terms in `k_K`, including the trainer's aggregate tail. This
surrogate has exactly the sample gradient of `mean(D_K²/(2*beta))` for the
implemented KL scalar. `stats["regression_loss"]` reports that squared residual.
Use `klpo_sequence_topk_loss(..., conditional_log_probs=..., behavior_conditional_log_probs=...)`.

### KLPO sequence regression + Binary KL

For sampled policy tokens, let `p` be the current trainer probability, `q` the
actual historical sampler probability, and `ell = log(p) - log(q)`. Define

```text
k_bin = q log(q/p) + (1-q) log((1-q)/(1-p))
D     = R - beta * sum_tokens(ell + k_bin)
omega = (1-q)/(1-p)
loss  = -mean_responses(stopgrad(D) * sum_tokens(stopgrad(omega) * log(p)))
```

This backward surrogate has the sample gradient of `mean(D² / (2*beta))`.
`stats["regression_loss"]` reports the latter; the returned loss is for backprop.
Both `D` and `omega` are recomputed after every learner update and detached.

```python
from klpo import klpo_sequence_loss

# logps, sampler_logps, action_mask: [B, T]; rewards: [B]
# logps retains the current model's autograd graph.
loss, stats = klpo_sequence_loss(logps, sampler_logps, rewards, action_mask, beta=0.1)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The complementary-probability factor `omega` is not `p/q`. Constant rewards and
single-response batches can still produce updates.

Binary KL approximates full conditional KL; it does not generally preserve the
exact canonical PMD stationary point or baseline invariance. The implementation
uses stable `log1mexp` arithmetic without probability smoothing. Compute model
`log_softmax` in float32 or higher before gathering sampled tokens. The binary
route rejects active log-probability zero: rounding a probability to one destroys
the complement needed by its formula. Half precision inputs are promoted, but
promotion cannot recover precision already lost in the model forward.

### Conditional records and tail stabilization

When selecting TopK-KL, store the sampler's original top-K IDs/probabilities (default K=128), and
gather the current trainer scores at those same IDs. Never renormalize the head,
reselect it using the trainer, or insert a sampled action that fell outside it.
Pass the sampled action's actual logps separately. Token regression's entire head coefficients
are detached; gradients flow through both current log-probability expressions.

Both TopK-KL APIs expose `tail_floor=1e-6` and report `floored_tokens`. For sequence regression,
`k_K` uses `q_tail_clamped * log(q_tail_clamped/p_tail_clamped)`; the literal
clamped KL is differentiated, so a trainer tail below the floor has zero
derivative through that clamp. This preserves the squared-residual gradient
identity for the stabilized scalar. Token regression retains the report's floored-ratio
backward approximation in `rho`, whose derivative need not equal that of the
clamped KL scalar once a floor is active. Flooring is an additional approximation
in either route. With inactive floors, both use the same coarsened-KL derivative.

```python
from klpo import klpo_sequence_topk_loss

loss, stats = klpo_sequence_topk_loss(  # For tokens: klpo_token_loss(..., kl_estimator="topk").
    logps, sampler_logps, rewards, action_mask, beta=0.1,
    conditional_log_probs=current_full_logps.gather(-1, stored_head_ids),
    behavior_conditional_log_probs=stored_head_logps,
)
```

For full conditionals, set `full_vocabulary=True` on either TopK-KL API
(token regression requires `kl_estimator="topk"`), or use `kl_estimator="full"`
on token regression: this uses the exact sum, without a tail
ratio or floor. Full-KL sequence regression and full-conditional token updates have
the same **population** gradient under the report's deterministic-transition,
terminal-reward, and matched-sampler assumptions. Individual sample gradients
can differ. Binary KL and finite-K KL are distinct approximations.

The TopK-KL API is `klpo_sequence_topk_loss` and the estimator name is `topk`.

## Training and optional KL budget

See [the training guide](training.md) for the R1/Qwen-Math Molt launchers,
installation, tensor contracts, and limitations. The native Molt integration supports
**both regression routes × MC-KL, TopK-KL, Binary KL, and Full KL** with actual
sampler records, raw terminal rewards, and global trajectory normalization.
It defaults to `--route token --kl-estimator mc --mc-samples 128`. Select
`--route sequence` for sequence regression, or `--kl-estimator topk|binary|full`
for another KL estimator. TopK-KL uses `--top-k 128` by default.
The fork collects auxiliary records and scores the same token IDs in the trainer;
KLPO supplies its mathematical loss through the backend's native interface.
No source patcher or old integration aliases are retained.

`predicted_kl` and `kl_budget_scale` implement optional Step 10. The toy example
forms a complete AdamW displacement, computes a logit JVP on fixed histories,
scales the displacement, and records realized KL. This is a prediction for
current-to-new trainer KL, not a guaranteed bound or a historical-sampler gate.
The Molt launcher leaves this optional step disabled.

## Validation

Tests cover all eight route/estimator combinations, exhaustive independent MC
gradient expectations, leave-one-out U-statistics, naive plug-in covariance bias,
M=1 token updates, duplicates, negative MC estimates, and direct squared-residual
gradients (including clamped TopK-KL tails), binary score corrections, population
equivalence on an enumerated variable-length tree, TopK-KL tail corrections, extreme probabilities,
masked NaN/inf, detached collection records, replay, complete optimizer budgets,
and microbatch/uneven-DP gradient normalization. An optional contract test checks
the native worker interface in the pinned Molt fork:

```bash
MOLT_SOURCE_PATH=/path/to/labs-molt python -m pytest -q
```

GPU training and paper-scale benchmark reproduction require a compatible Linux
CUDA environment; the local verification is CPU-based. `beta=0.1` and the
inherited model-training hyperparameters are starting values, not tuned KLPO
benchmark settings.
