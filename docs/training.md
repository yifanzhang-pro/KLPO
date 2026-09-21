# Training KLPO

[Project home](../README.md) · [Algorithm reference](algorithms.md) · [Paper](../KLPO.pdf)

## Native Molt integration

The [labs-molt fork](https://github.com/yifanzhang-pro/labs-molt/tree/feat/klpo-all-kl)
supports all eight combinations of `--route token|sequence` and
`--kl-estimator mc|topk|binary|full`. The default is **token regression + MC-KL**,
with M=128 auxiliary draws per prefix. Selecting sequence regression also uses
MC-KL unless another estimator is specified. TopK-KL remains optional, with K=128.
Molt owns generation-time probability capture, replay transport, differentiable
trainer scoring, FSDP2, accumulation, evaluation, and checkpoints. Its native
`actor.loss_mode=klpo` invokes `klpo.molt.KLPOLoss` for the regression formulas.
There is no source transformation or runtime monkey patch.

In a Linux NVIDIA GPU environment, install the pinned fork and KLPO into the
same environment on every Ray worker:

```bash
git clone --branch feat/klpo-all-kl https://github.com/yifanzhang-pro/labs-molt.git labs-molt-klpo
git -C labs-molt-klpo checkout --detach e24e22faaf0d3cf3ad56dec47215117d95914be2
export MOLT_PATH="$PWD/labs-molt-klpo"
export KLPO_PATH=/absolute/path/to/KLPO
pip install -e "$MOLT_PATH"
pip install -e "$KLPO_PATH"
python "$KLPO_PATH/scripts/check_molt.py" --molt-path "$MOLT_PATH"
```

`check_molt.py` is read-only. It checks the source revision, native API version,
and tracked-file cleanliness. The launcher repeats this check before starting.
Use this clean native checkout instead of an older source-patched backend.

| Launcher option | Native Molt option | Meaning |
| --- | --- | --- |
| `--route token|sequence` | `--actor.klpo_route` | Regression coefficient; default: token |
| `--kl-estimator mc|topk|binary|full` | `--actor.klpo_kl_estimator` | Conditional KL estimator; default: MC-KL |
| `--mc-samples 128` | `--actor.klpo_mc_samples` | Default MC-KL: IID draws M per prefix, with replacement |
| `--top-k 128` | `--actor.klpo_top_k` | Optional TopK-KL: head size K, capped at vocabulary size |
| `--beta 0.1` | `--actor.klpo_beta` | Regularization strength |
| `--tail-floor 1e-6` | `--actor.klpo_tail_floor` | TopK-KL tail stabilization |

Token MC-KL allows M=1; sequence MC-KL requires M >= 2 for independent
leave-one-out residuals. M counts auxiliary tokens, not full response rollouts.

### R1 math

```bash
python "$MOLT_PATH/examples/python/utils/prepare_dapo.py" \
  --train-source sail/Sanity-Test-R1D-1.5B \
  --eval-source sail/Sanity-Test-R1D-1.5B --eval-split test \
  --out-dir "$PWD/data/sanity-r1d"
ray start --head --num-gpus=8 --disable-usage-stats
python "$KLPO_PATH/scripts/train_molt.py" \
  --recipe r1 --molt-path "$MOLT_PATH" \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --train-data "$PWD/data/sanity-r1d/train" \
  --eval-data "$PWD/data/sanity-r1d/eval" \
  --beta 0.1 --output "$PWD/outputs/klpo-r1" --dry-run
# Remove --dry-run to train.
# Default: token regression + MC-KL, M=128.
# Add --route sequence for sequence regression; --kl-estimator topk --top-k 128 for TopK-KL.
```

### Qwen2.5-Math

```bash
python "$KLPO_PATH/scripts/train_molt.py" \
  --recipe qwen_math --molt-path "$MOLT_PATH" \
  --model /path/to/Qwen2.5-Math-1.5B \
  --train-data /path/to/deduplicated-7500-dapo/train \
  --eval-data /path/to/five-benchmark-eval \
  --beta 0.1 --output "$PWD/outputs/klpo-qwen-math" --dry-run
```

Default placement is one actor GPU and seven rollout GPUs. Adjust `--actor-gpus`
and `--rollout-gpus`; actor GPUs must divide the batch of 128. `--episodes` counts
dataset passes, not optimizer updates. Training temperature and top-p are 1.
The recipe collects one response per prompt and performs one optimizer update
per complete batch. `force_sync_mode`, queue depth one, and a drained rollout
batch prevent requests from crossing the weight refit. Auxiliary MC records are
used once, before any update can depend on them. Evaluation collects no auxiliary
records and may use its own temperature/top-p.

The `/molt/v1/generate` endpoint captures probabilities directly from vLLM's
processed generation logits. TopK-KL preserves the K highest-probability token IDs
without renormalizing or inserting the realized action. MC uses a separate CPU
RNG to draw with replacement from that prefix's full distribution, preserving
repeated IDs; it never generates additional response continuations. Full KL
stores the entire distribution in vocabulary-ID order. Missing records fail
rather than falling back to a recomputed trainer distribution.

**Cost:** TopK-KL transfers O(TK) conditional records. Full transfers O(TV). The current
MC capture also materializes full sampler logprobs internally before reducing
them to O(TM) records on the engine host. It saves transport/replay storage, but
is not yet a fused O(M) GPU sampler. Long-context Full/MC collection can be costly.
Full trainer scoring materializes vocabulary probabilities; selected-ID scoring
uses chunked normalization. TP gathers vocabulary logits for auxiliary scoring.

KLPO always uses actual `experience.rollout_log_probs`, never PPO's old-policy
scoring pass. `experience.kl_log_probs` and `kl_token_ids` store conditional
records as `[B,K/M/V,T-1]` for sequence-last padding; the worker presents them to
the loss as `[B,T-1,K/M/V]`. Full KL omits the redundant token IDs. Packing and
context-parallel layouts restore the full response axis before regression.
Raw `experience.rewards` enters the loss separately from advantage telemetry.

Sequence regression reports its squared residual, or the unbiased U-statistic
estimate for MC-KL (which may be negative). Token regression reports its detached
backward surrogate. Both report the chosen conditional KL per policy token.
Loss telemetry is averaged by response, and gradients use the global number of
responses over all DP ranks and microbatches, without a second accumulation divisor.

Training requires positive temperature, top-p=1, no top-k sampling truncation,
penalties, or minimum-token EOS suppression. Native KLPO currently rejects
partial rollouts, MTP speculative decoding, routing replay, and context-compacted
response segments. Multi-turn responses retain their tool/context masks and must
remain a single complete trajectory. These checks keep the sampler/trainer
probability contract explicit. The local validation is CPU-based; CUDA/vLLM
training and multi-GPU performance have not been run on this machine.

The optimizer settings are inherited starting points: AdamW lr=1e-6, weight
decay=.1, gradient norm limit=1, cosine schedule, warmup=.03. The gradient norm
limit acts on the optimizer proposal; there is no policy-ratio clipping, binary
probability clamp, trust gate, or dynamic outcome filter in the KLPO objective.
Retune learning rate and beta for the trajectory-sum objective: it has a different
scale from FlashREINFORCE's sample-mean objective.

These launchers are single-turn math recipes (`MAX_AGENT_TURNS=1`). A future
multi-turn adapter must assemble all turns of a rollout into one loss row. Do not
train on per-turn fragments using the complete-rollout terminal reward. Stochastic
tools also require checking the assumptions behind the exact sequence-regression identity.

## Optional token and sequence regression combinations

```bash
python examples/train_toy.py --route token --kl-estimator binary
python examples/train_toy.py --route sequence --kl-estimator topk --top-k 2
```

The first uses `klpo_token_loss(..., kl_estimator="binary")`:
`-sum(sg(R-beta*ell) * sg((1-q)/(1-p)) * log(p))`, averaged over trajectories.
No head or full-vocabulary records are needed. This is an approximate score
correction and does not inherit the exact full-KL population identity in general.

The second uses `klpo_sequence_topk_loss`: it computes a KL over the stored K head actions
and one aggregate tail, then uses that KL in both the dynamic trajectory residual
`D_K = R - beta*sum(ell+k_K)` and the differentiable correction. For a sampled
action outside the head, its logps still enter `ell` independently. Never insert
that sampled action into the stored head or recompute the head from the trainer.

For numerical stability, sequence regression clamps both aggregate tail masses to `tail_floor`
in its KL scalar and differentiates that literal scalar. This makes the backward
surrogate match `D_K²/(2*beta)` even when a floor is active; the derivative through
a clamped trainer tail is zero. KLPO token regression + TopK-KL retains its existing floored-ratio
backward approximation. The two tail corrections coincide when floors are
inactive; `floored_tokens` identifies where stabilization affects them.

For K=V, both TopK-KL APIs require `full_vocabulary=True` (token regression requires
`kl_estimator="topk"`, or instead accepts `kl_estimator="full"`); they bypass the aggregate tail and recover their full-KL
routes. The toy CLI detects K=V automatically. With K=V-1 and inactive floors,
the singleton tail also recovers full KL. A singleton head containing the sampled
action instead recovers the binary partition, again when floors are inactive.

## Tensor and reduction contracts

### MC-KL collection

MC-KL is the default in the CPU example and Molt launcher; use `--mc-samples M`
to set its sample count. K (`--top-k`)
controls only TopK-KL head size; M controls independent token draws, with no
vocabulary-size cap. Token regression permits M >= 1; sequence regression
requires M >= 2 and uses leave-one-out residuals to remove covariance bias.

For each complete rollout, draw M IID auxiliary actions **with replacement**
at every policy prefix from its actual historical sampler. These draws must
be independent of the rollout action/continuation, one another, and draws at
other prefixes. They do not execute tools or generate response continuations.
Store `[B,T,M]` IDs and original sampler logps; gather trainer logps at the
same IDs. Use the distinct `mc_log_probs` / `behavior_mc_log_probs` API fields,
not TopK-KL conditional-head fields. Duplicates are required outcomes of IID
sampling, and their probability sum need not be <= 1.

Resample from the historical conditionals at each update for a fresh MC estimate
independent of the current trainer. If only fixed MC records were stored and
the old sampler is unavailable, replay is an empirical-surrogate update, not
a fresh conditional-unbiased gradient. The CPU example keeps historical full
conditionals to demonstrate resampling; the native Molt recipe instead consumes each MC bank once.
MC-KL needs a sampler capable of drawing these additional actions and logging
their actual probabilities, even though it requires only one complete rollout.

MC KL and the sequence U-statistic loss estimate may be negative. Do not clamp
either. Rewards and sampler logps are detached, while both current rollout and
auxiliary-token logps must retain the trainer graph. Full-KL gradient identities
apply in expectation over independent auxiliary draws, with the original route
assumptions. Finite-M noise does not guarantee a stable or monotone sample update.

### Shared contracts

- Every loss row is a complete response with at least one policy token. Action
  masks exclude prompt, external observations, tool results, and padding.
- Compute the current model's log-softmax in float32 or higher. Rewards and all
  historical sampler records are constants for differentiation. Masked entries
  may be arbitrary; active sampled logps must be finite. Binary KL additionally
  requires strict negativity; no smoothing is silently applied.
- The library returns a local response mean of a token sum. For accumulation or
  data parallelism, multiply by `B_local * dp_size / B_global`. `B_global` counts
  trajectories across the entire optimizer step, not tokens or microbatches.
  Do not average this loss again by accumulation steps. Molt's native worker
  supplies its global sequence count and compensates its gradient averaging.
- Never split a trajectory across calls: its dynamic residual depends on all its
  policy tokens. Context-parallel forwards must restore the complete token axis
  before computing the loss.
- Store the sampler version and actual probabilities used to collect each
  trajectory, including sampling transforms. Publishing checkpoints must not
  replace historical records. On replay, recompute current scores and feedback.
  The CPU example demonstrates reuse; the provided GPU launcher uses one pass.
- Sampler full-vocabulary records may contain `-inf` for zero-mass actions; active
  sampled actions must have positive sampler mass. The full trainer support must
  be positive. Truncating sampler support or selecting trajectories by outcome
  can change the target and invalidates unqualified population claims.

## Optional Step 10

Given the optimizer's complete proposed parameter displacement `D`, evaluate
`J_logits @ D` on fixed generated-token probe histories. Pass current logits and
this JVP to `predicted_kl`, then call `kl_budget_scale(prediction, budget)` and
apply `theta_new = theta + alpha * D`. Include learning rate, momentum,
preconditioning and weight decay in D; scaling only raw gradients is insufficient.
Keep probe histories, model stochasticity, and the smooth temperature-scaled
softmax definition fixed. Budget calibration does not model quantization or
sampling truncation.

Zero budgets and missing, zero, negative or nonfinite predictions produce scale
zero. Flag invalid probes. Disabled calibration (`budget=None`) returns scale
one. After the update, log realized current-to-new probe KL: the quadratic
prediction is not a hard bound, trajectory KL bound, or sampler-to-trainer bound.
For distributed probes, reduce vocabulary-variance sums and probe-token counts
across ranks before choosing a common scale. The provided helper is local.

`examples/train_toy.py --kl-budget 1e-4` demonstrates the complete sequence with
AdamW. It retains the optimizer's moment update even when parameter displacement
is reduced or skipped. These primitives are not wired into the distributed Molt
optimizer; Step 10 is optional and disabled in that launcher.

## Verified scope

CPU tests exercise loss and optimizer gradients, numerical boundaries, repeated
updates, Top-K heads (including out-of-head sampled actions), full-vocabulary
limits, microbatch/DP scaling, launcher commands, and the pinned native backend
interface. This does not validate a CUDA, Ray, FSDP, vLLM, or multi-node training
run. No paper-scale result is claimed.
