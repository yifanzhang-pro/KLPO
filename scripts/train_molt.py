"""Launch KLPO sequence/token regression through the labs-molt fork native KL interface.

Recipe scaffolding adapted from yifanzhang-pro/FlashREINFORCE (Apache-2.0).
"""

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_molt import MOLT_REVISION, verify_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["r1", "qwen_math"], default="r1")
    parser.add_argument("--route", choices=["sequence", "token"], default="token",
                        help="KLPO sequence/token regression")
    parser.add_argument("--kl-estimator", choices=["binary", "topk", "mc", "full"], default="mc")
    parser.add_argument("--top-k", type=int, default=128, help="TopK-KL head size K")
    parser.add_argument("--mc-samples", type=int, default=128, help="Independent MC samples M per prefix")
    parser.add_argument("--tail-floor", type=float, default=1e-6)
    parser.add_argument("--molt-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/klpo"))
    parser.add_argument("--actor-gpus", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Dataset passes, NOT optimizer updates")
    parser.add_argument("--beta", type=float, default=0.1, help="Positive KLPO regularization coefficient")
    parser.add_argument("--attention", default="flash_attention_2", choices=["flash_attention_2", "te"])
    parser.add_argument("--dry-run", action="store_true", help="Print command without importing Molt or starting training")
    args = parser.parse_args()
    if min(args.actor_gpus, args.rollout_gpus, args.episodes) < 1 or 128 % args.actor_gpus:
        parser.error("GPU counts/episodes must be positive; actor GPUs must divide batch size 128")
    r1 = args.recipe == "r1"
    if not math.isfinite(args.beta) or args.beta <= 0:
        parser.error("beta must be positive and finite")
    if args.top_k < 1 or args.mc_samples < 1 or not 0 < args.tail_floor < 1:
        parser.error("K/M must be positive; tail-floor must be in (0, 1)")
    if args.route == "sequence" and args.kl_estimator == "mc" and args.mc_samples < 2:
        parser.error("Sequence MC-KL requires M >= 2 for the independent cross estimator")
    root = args.molt_path.expanduser().resolve()
    agent = root / "examples/python/agents/math.py"
    output = args.output.expanduser().resolve()
    flags = {
        "actor.model_name_or_path": args.model,
        "data.prompt_dataset": args.train_data.expanduser().resolve(),
        "data.input_key": "prompt", "data.label_key": "reward_model",
        "data.max_samples": 1460 if r1 else 7500,
        "data.max_len": 9216 if r1 else 6144,
        "rollout.batch_size": 128, "rollout.vllm_generate_batch_size": 128,
        "rollout.micro_batch_size": 1, "rollout.n_samples_per_prompt": 1,
        "rollout.max_new_tokens": 8192 if r1 else 4096,
        "rollout.temperature": 1.0, "rollout.top_p": 1.0,
        "train.batch_size": 128, "train.micro_batch_size": 1,
        "train.max_epochs": 1, "train.num_episodes": args.episodes,
        "train.async_queue_size": 1,
        "actor.num_nodes": 1, "actor.num_gpus_per_node": args.actor_gpus,
        "ref.num_nodes": 1, "ref.num_gpus_per_node": args.actor_gpus,
        "vllm.num_engines": args.rollout_gpus, "vllm.tensor_parallel_size": 1,
        "vllm.sync_backend": "nccl", "vllm.gpu_memory_utilization": 0.9,
        "fsdp.param_dtype": "bf16", "fsdp.attn_implementation": args.attention,
        "actor.gradient_checkpoint": "full", "actor.optim": "adam",
        "actor.adam.lr": 1e-6, "actor.adam.weight_decay": 0.1,
        "actor.lr_scheduler": "cosine_with_min_lr", "actor.lr_warmup_ratio": 0.03,
        "actor.min_lr_ratio": 0.1, "actor.max_norm": 1.0,
        "actor.loss_mode": "klpo", "actor.klpo_beta": args.beta,
        "actor.klpo_route": args.route,
        "actor.klpo_kl_estimator": args.kl_estimator,
        "actor.klpo_top_k": args.top_k, "actor.klpo_mc_samples": args.mc_samples,
        "actor.klpo_tail_floor": args.tail_floor, "actor.entropy_coef": 0,
        "algo.advantage.estimator": "reinforce",
        "algo.advantage.gamma": 1,
        "algo.advantage.is_correction_level": "off", "algo.kl.init_coef": 0,
        "train.agent_path": agent, "eval.dataset": args.eval_data.expanduser().resolve(),
        "eval.steps": 128 if r1 else 100,
        "eval.n_samples_per_prompt": 32 if r1 else 16,
        "eval.temperature": 0.6 if r1 else 0.7, "eval.top_p": 0.95 if r1 else 0.7,
        "ckpt.output_dir": output / "hf", "ckpt.path": output / "state",
        "ckpt.save_steps": 50, "logger.logging_steps": 1,
    }
    command = [sys.executable, "-u", "-m", "molt.cli.train_rl_ray"]
    for key, value in flags.items():
        command.extend([f"--{key}", str(value)])
    command += ["--data.apply_chat_template", "--train.force_on_policy", "--train.force_sync_mode",
                "--train.colocate_fsdp_models", "--fsdp.packing_samples", "--eval.eval_at_start",
                "--algo.advantage.no_whiten"]
    print(f"Molt revision: {MOLT_REVISION}\nWorking directory: {root}", flush=True)
    print("MAX_AGENT_TURNS=1 " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not agent.is_file():
        parser.error(f"Molt math agent not found: {agent}")
    try:
        verify_backend(root)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    try:
        import klpo
    except ImportError:
        parser.error("Install this repository into the Molt training environment: pip install -e /path/to/KLPO")
    expected_package = Path(__file__).resolve().parents[1] / "klpo"
    if Path(klpo.__file__).resolve().parent != expected_package:
        parser.error("The imported klpo package is not this checkout; reinstall this repository")
    for dataset in (args.train_data, args.eval_data):
        if not dataset.expanduser().exists():
            parser.error(f"Prepared dataset does not exist: {dataset}")
    env = os.environ.copy()
    env.update(MAX_AGENT_TURNS="1", TOKENIZERS_PARALLELISM="true", RAY_USAGE_STATS_ENABLED="0",
               VLLM_WORKER_MULTIPROC_METHOD="spawn")
    # Ray's lifecycle belongs to the caller; do not stop a shared cluster here.
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
