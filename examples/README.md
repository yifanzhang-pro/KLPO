# Single-turn and multi-turn KLPO experiments

These recipes adapt all five base settings in [FlashREINFORCE's examples](https://github.com/yifanzhang-pro/FlashREINFORCE/tree/a1ebe22/examples) to KLPO's native Molt interface: two single-turn math experiments and three multi-turn experiments. **Token regression + MC-KL, M=128, is the default.** Each training prompt produces one complete trajectory, including all assistant turns and tool observations. Both regression routes and all four KL estimators are selectable.

Single-turn math uses Molt's built-in math agent. For multi-turn experiments, supply a compatible Python-tool or ALFWorld agent. Prepared data, checkpoints, and GPU placement are supplied separately. These are experiment starting points; GPU runs and benchmark results are not included. The existing `scripts/train_molt.py --recipe r1|qwen_math --sync` entry point is also retained.

## Settings

| Setting | Model | Max turns | Context / per-turn generation cap | Training prompts × rollouts | Dataset passes | Evaluation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `deepseek_r1_distill_qwen_1p5b_math_sanity` | DeepSeek-R1-Distill-Qwen-1.5B | 1 | 9,216 / 8,192 | 128 × 1 | 350 | Every 128 updates, 32 samples/prompt |
| `qwen2p5_math_1p5b_dapo_math` | Qwen2.5-Math-1.5B | 1 | 6,144 / 4,096 | 128 × 1 | 68 | Every 100 updates, 4 samples/prompt |
| `qwen2p5_7b_instruct_python_tool_10turn` | Qwen2.5-7B-Instruct | 10 | 8,192 / 6,144 | 128 × 1 | 68 | Every 100 updates, 4 samples/prompt |
| `qwen3_30b_a3b_python_tool_20turn` | Qwen3-30B-A3B | 20 | 16,384 / 14,336 | 128 × 1 | 17 | Every 100 updates, 4 samples/prompt |
| `qwen2p5_7b_instruct_alfworld_50turn` | Qwen2.5-7B-Instruct | 50 | 16,384 / 8,192 | 64 × 1 | 20 | Every 40 updates, 3 samples/game |

R1 selects up to 1,460 training prompts; Qwen-Math and Python-tool settings select up to 7,500; ALFWorld selects up to 1,024 games. `--episodes` overrides dataset passes, **not** optimizer updates. Context includes the prompt, all generated tokens, and tool observations. The generation cap applies to each call and is also bounded by the remaining context; reaching the context budget can end a trajectory before its turn limit. Evaluation runs at step zero as well as at the stated interval.

Model names, budgets, optimizer settings, and evaluation schedules come from the source. KLPO uses its own trajectory-sum loss and needs learning-rate/beta tuning. FlashREINFORCE's negative-token selection, trust-gate, reduction, and GRPO ablations are not KLPO options. The Qwen3 setting adopts the source's **20-turn** budget; it is not its trust-gate ablation or its separate 10-turn paper comparison.

## Preview and launch

Inspect a merged setting with just Python; neither Torch nor Molt is imported:

```bash
python examples/run_experiment.py \
  --setting deepseek_r1_distill_qwen_1p5b_math_sanity --print-config
python examples/run_experiment.py \
  --setting qwen2p5_7b_instruct_python_tool_10turn --print-config
```

For training, follow the [pinned Molt installation](../docs/training.md#native-molt-integration), installing Molt, KLPO, and your agent's dependencies on every worker. Set `MOLT_PATH` and `KLPO_PATH` as in that guide, and start or connect to a Ray cluster with enough GPUs. The launcher leaves the shared cluster running.

### Single-turn math

These settings automatically select `$MOLT_PATH/examples/python/agents/math.py` and one assistant turn. No `--agent-path` is needed; it remains available for a compatible custom grader.

```bash
python "$KLPO_PATH/examples/run_experiment.py" \
  --setting deepseek_r1_distill_qwen_1p5b_math_sanity \
  --molt-path "$MOLT_PATH" \
  --train-data /path/to/sanity-r1d/train --eval-data /path/to/aime24-25/eval \
  --actor-gpus 1 --rollout-engines 7 \
  --output "$PWD/outputs/klpo-r1-singleturn" --dry-run

python "$KLPO_PATH/examples/run_experiment.py" \
  --setting qwen2p5_math_1p5b_dapo_math \
  --molt-path "$MOLT_PATH" \
  --train-data /path/to/deduplicated-7500-dapo/train \
  --eval-data /path/to/amc23-minerva-aime25/eval \
  --actor-gpus 1 --rollout-engines 7 \
  --output "$PWD/outputs/klpo-qwen-math-singleturn" --dry-run
```

The new JSON settings follow the source's 350/68 dataset passes and avg@32/avg@4 evaluation. The existing `scripts/train_molt.py` keeps its original defaults (1,000 passes and avg@32/avg@16); choose the intended protocol explicitly when comparing runs. The [training guide](../docs/training.md) retains those commands and dataset preparation instructions.

### Multi-turn Python tools and ALFWorld

The following 4-actor/4-rollout allocation is illustrative; choose GPU memory and parallelism for your model and context:

```bash
python "$KLPO_PATH/examples/run_experiment.py" \
  --setting qwen2p5_7b_instruct_python_tool_10turn \
  --molt-path "$MOLT_PATH" --agent-path /path/to/python_tool_agent.py \
  --train-data /path/to/tool-dapo-7500/train \
  --eval-data /path/to/tool-math/eval \
  --actor-gpus 4 --rollout-engines 4 \
  --output "$PWD/outputs/klpo-python-10turn" --dry-run
```

For Qwen3, select `--setting qwen3_30b_a3b_python_tool_20turn` and a Qwen3-compatible tool agent. Use `--actor-nodes`, `--actor-gpus` (per node), `--actor-tp`, `--actor-ep`, `--actor-cp`, `--rollout-engines`, and `--rollout-tp` to fit the MoE model. `--model` can select a local checkpoint instead of the setting's Hugging Face ID.

```bash
export ALFWORLD_DATA=/absolute/path/to/alfworld-data
python "$KLPO_PATH/examples/run_experiment.py" \
  --setting qwen2p5_7b_instruct_alfworld_50turn \
  --molt-path "$MOLT_PATH" --agent-path /path/to/alfworld_agent.py \
  --train-data /path/to/alfworld/train --eval-data /path/to/alfworld/eval \
  --actor-gpus 4 --rollout-engines 4 \
  --output "$PWD/outputs/klpo-alfworld-50turn" --dry-run
```

Remove `--dry-run` to execute. Preview does not check dependencies, data, agent behavior, or memory capacity. A real launch verifies the pinned backend and installed KLPO checkout, checks the trainer's flags, and saves `resolved_settings.json` alongside `hf/` and `state/`. Use a distinct output directory per experiment. The saved JSON includes model, source recipe, backend revision, actual flags, and agent environment settings.

## Regression and KL comparisons

Append these options to any launch command; compare runs with the same data, agent, model, and schedule:

| Regression route | MC-KL | TopK-KL | Binary KL | Full KL |
| --- | --- | --- | --- | --- |
| **Token (default)** | No extra flags; `--mc-samples 128` | `--kl-estimator topk --top-k 128` | `--kl-estimator binary` | `--kl-estimator full` |
| Sequence | `--route sequence --mc-samples 128` | `--route sequence --kl-estimator topk --top-k 128` | `--route sequence --kl-estimator binary` | `--route sequence --kl-estimator full` |

Use `--beta` to set regularization. M counts IID auxiliary token draws per policy prefix (token regression M ≥ 1; sequence regression M ≥ 2), not complete rollouts. K is the TopK-KL head size. The current MC implementation captures full sampler logprobs internally before reducing transport to M records; Full KL retains the whole vocabulary. Both can be expensive at these context lengths. See [sampling and tensor contracts](../docs/training.md#tensor-and-reduction-contracts).

## Agent and dataset contract

- Single-turn math datasets use `prompt` chat messages, `reward_model` ground-truth labels, and `datasource` benchmark labels. The built-in agent grades the final boxed answer and ends after one response. Prepare AIME24/25 for R1 avg@32 and AMC23/Minerva/AIME25 for Qwen-Math avg@4; verify the Qwen-Math checkpoint supports a 6,144-token context.
- Use a Molt `Env` with `AgentRunner(StepEnvRunner)`, or a compatible `ChatAgent`/runner. The agent must read `MAX_AGENT_TURNS`, count assistant turns, and terminate or truncate at that limit. The launcher forwards this value to Ray workers; it cannot enforce an arbitrary agent's loop. Prefer `StepEnvRunner`, which preserves action spans and native KL records across calls.
- Return zero reward on intermediate tool steps and the terminal reward once. Python-tool reward should grade only the committed final answer; ALFWorld reward should represent terminal task success. Close environment/executor resources after each trajectory and define failure/timeout rewards explicitly.
- Return **one complete trajectory per prompt**. Assistant tokens from all turns contribute to the same loss row; prompts, tool outputs, observations, and padding have a false action mask. Preserve the actual sampler logprobs and auxiliary records at their original positions. Do not reconstruct them from retokenized text, split turns into separate rewarded samples, or compact history during a rollout.
- Keep training generation full-support: temperature=1, top-p=1, no top-k truncation or penalties, and preserve native `molt_kl` sampling metadata across turns. Evaluation may use its configured temperature/top-p. The agent must not resample or overwrite the auxiliary MC draws.
- Python-tool datasets use `prompt` chat messages and `reward_model` labels understood by the agent's grader. Include tool instructions/schema and choose a parser/executor matching the model's output format. Supply `datasource` labels to distinguish evaluation benchmarks. The source's 10-turn protocol evaluates AMC23, Minerva, and AIME25 with four samples per prompt; prepare these inputs separately.
- ALFWorld datasets use `prompt` and `game_file`. Install ALFWorld and game data on every worker. `ALFWORLD_DATA` and every game-file path must resolve there; the launcher forwards the absolute data directory but does not copy data. For the source protocol, provide the 140 seen and 134 unseen evaluation games, with distinct `datasource` labels and three samples per game.

The pinned backend requires synchronous collection, queue depth one, complete batches, and one optimizer update per batch. These recipes set both `force_on_policy` and `force_sync_mode`, and make generation, rollout, and training batch sizes equal. The R1 and Qwen3 source async queues are therefore changed to one; R1's concurrent generation batch is reduced from 512 to 128. This is an execution restriction of the current integration, not a restriction of the KLPO loss API. Stochastic environments also require checking the assumptions behind exact sequence-regression identities.

## Validation

```bash
MOLT_SOURCE_PATH="$MOLT_PATH" python -m pytest -q tests/test_multiturn.py
```

CPU tests execute the pinned trainer's argument parser and validation for all 40 setting/route/estimator combinations, check the single-turn default agent and required multi-turn agents, exercise the launch wrapper with a fake Ray backend, and check that tool-observation masks preserve complete-trajectory losses and gradients for all eight loss combinations. They do not run an actual tool environment, CUDA training, or a benchmark. Record seeds, dataset revisions, agent revision, checkpoint, and GPU layout with each real experiment before reporting results.
