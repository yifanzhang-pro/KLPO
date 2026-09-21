"""Launch single-turn and multi-turn KLPO experiments on the pinned Molt backend.

Experiment scaffolding adapted from FlashREINFORCE (Apache-2.0); see NOTICE.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_molt import MOLT_REVISION, verify_backend

SETTINGS = Path(__file__).resolve().parent / "settings"
SETTING_NAMES = sorted(path.stem for path in SETTINGS.glob("*.json")
                       if path.name != "klpo_shared_defaults.json")


def load_settings(name):
    if name not in SETTING_NAMES:
        raise ValueError(f"Unknown experiment setting: {name}")
    config = json.loads((SETTINGS / f"{name}.json").read_text())
    defaults = json.loads((SETTINGS / "klpo_shared_defaults.json").read_text())
    config["flags"] = defaults | config["flags"]
    return config


def flag_arguments(flags):
    arguments = []
    for key, value in flags.items():
        if value is False or value is None:
            continue
        arguments.append(f"--{key}")
        if value is not True:
            arguments.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=SETTING_NAMES, required=True)
    parser.add_argument("--route", choices=["token", "sequence"], default="token")
    parser.add_argument("--kl-estimator", choices=["mc", "topk", "binary", "full"], default="mc")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--tail-floor", type=float, default=1e-6)
    parser.add_argument("--molt-path", type=Path)
    parser.add_argument("--agent-path", type=Path,
                        help="Molt Env/ChatAgent; math settings default to Molt's single-turn math agent")
    parser.add_argument("--model", help="Override the setting's model with a checkpoint or HF ID")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/klpo-experiments"))
    parser.add_argument("--actor-nodes", type=int, default=1)
    parser.add_argument("--actor-gpus", type=int, help="Actor GPUs per node")
    parser.add_argument("--rollout-engines", type=int)
    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--actor-tp", type=int, default=1)
    parser.add_argument("--actor-ep", type=int, default=1)
    parser.add_argument("--actor-cp", type=int, default=1)
    parser.add_argument("--attention", choices=["flash_attention_2", "te"], default="flash_attention_2")
    parser.add_argument("--episodes", type=int, help="Override dataset passes, not optimizer updates")
    parser.add_argument("--print-config", action="store_true", help="Print merged settings without GPU dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Preview the command without starting training")
    args = parser.parse_args()
    if not math.isfinite(args.beta) or args.beta <= 0:
        parser.error("beta must be positive and finite")
    if args.mc_samples < 1 or args.top_k < 1 or not 0 < args.tail_floor < 1:
        parser.error("K/M must be positive; tail-floor must be in (0, 1)")
    if args.route == "sequence" and args.kl_estimator == "mc" and args.mc_samples < 2:
        parser.error("Sequence MC-KL requires M >= 2 for independent leave-one-out residuals")
    if args.episodes is not None and args.episodes < 1:
        parser.error("episodes must be positive")
    config = load_settings(args.setting)
    flags = config["flags"]
    flags.update({
        "actor.klpo_route": args.route, "actor.klpo_kl_estimator": args.kl_estimator,
        "actor.klpo_beta": args.beta, "actor.klpo_mc_samples": args.mc_samples,
        "actor.klpo_top_k": args.top_k, "actor.klpo_tail_floor": args.tail_floor,
    })
    if args.episodes is not None:
        flags["train.num_episodes"] = args.episodes
    if args.model is not None:
        config["model"] = args.model
    config["molt_revision"] = MOLT_REVISION
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    for required in ("molt_path", "train_data", "eval_data", "actor_gpus", "rollout_engines"):
        if getattr(args, required) is None:
            parser.error(f'--{required.replace("_", "-")} is required to build a command')
    if min(args.actor_nodes, args.actor_gpus, args.rollout_engines, args.rollout_tp,
           args.actor_tp, args.actor_ep, args.actor_cp) < 1:
        parser.error("GPU and parallelism counts must be positive")
    root = args.molt_path.expanduser().resolve()
    if args.agent_path is None and config["agent_kind"] != "math":
        parser.error("--agent-path is required for Python-tool and ALFWorld settings")
    agent = (args.agent_path.expanduser().resolve() if args.agent_path is not None
             else root / "examples/python/agents/math.py")
    output = args.output.expanduser().resolve()
    flags.update({
        "actor.model_name_or_path": config["model"], "train.agent_path": str(agent),
        "data.prompt_dataset": str(args.train_data.expanduser().resolve()),
        "eval.dataset": str(args.eval_data.expanduser().resolve()),
        "actor.num_nodes": args.actor_nodes, "actor.num_gpus_per_node": args.actor_gpus,
        "ref.num_nodes": args.actor_nodes, "ref.num_gpus_per_node": args.actor_gpus,
        "fsdp.tp_size": args.actor_tp, "fsdp.ep_size": args.actor_ep, "fsdp.cp_size": args.actor_cp,
        "vllm.num_engines": args.rollout_engines, "vllm.tensor_parallel_size": args.rollout_tp,
        "fsdp.attn_implementation": args.attention,
        "ckpt.output_dir": str(output / "hf"), "ckpt.path": str(output / "state"),
    })
    command = [sys.executable, "-u", str(ROOT / "scripts/run_molt_experiment.py"),
               "--max-agent-turns", str(config["max_agent_turns"]), "--", *flag_arguments(flags)]
    print(f"Molt revision: {MOLT_REVISION}\nWorking directory: {root}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print("Preview only; backend, agent, datasets, and GPU placement have not been checked.")
        return
    for path in (agent, Path(flags["data.prompt_dataset"]), Path(flags["eval.dataset"])):
        if not path.exists():
            parser.error(f"Required agent or dataset not found: {path}")
    if config["agent_kind"] == "alfworld":
        data = os.environ.get("ALFWORLD_DATA")
        if not data or not Path(data).is_dir():
            parser.error("Set ALFWORLD_DATA to a game-data directory available at the same path on all workers")
    try:
        verify_backend(root)
        import klpo
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as exc:
        parser.error(f"Install the pinned Molt and this KLPO checkout in the training environment: {exc}")
    if Path(klpo.__file__).resolve().parent != ROOT / "klpo":
        parser.error("The imported klpo package is not this checkout; reinstall this repository")
    env = os.environ.copy()
    env.update(TOKENIZERS_PARALLELISM="true", RAY_USAGE_STATS_ENABLED="0", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    if config["agent_kind"] == "alfworld":
        env["ALFWORLD_DATA"] = str(Path(env["ALFWORLD_DATA"]).resolve())
    result = subprocess.run([sys.executable, "-m", "molt.cli.train_rl_ray", "--help"],
                            cwd=root, env=env, capture_output=True, text=True)
    if result.returncode:
        parser.error("Trainer capability check failed:\n" + result.stderr[-3000:])
    available = set(re.findall(r"--[A-Za-z0-9_.-]+", result.stdout))
    missing = sorted(f"--{key}" for key, value in flags.items()
                     if value is not False and value is not None and f"--{key}" not in available)
    if missing:
        parser.error("Pinned trainer does not support: " + ", ".join(missing))
    config["worker_env"] = {"MAX_AGENT_TURNS": str(config["max_agent_turns"])}
    if env.get("ALFWORLD_DATA"):
        config["worker_env"]["ALFWORLD_DATA"] = env["ALFWORLD_DATA"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_settings.json").write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
