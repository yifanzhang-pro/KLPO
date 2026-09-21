"""CPU checks for complete multi-turn trajectories and the experiment entry point."""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import pytest
import torch

from klpo.molt import KLPOLoss

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("multiturn_experiments", ROOT / "examples/run_experiment.py")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)


def launch_arguments(tmp_path, setting=None):
    return ["--setting", setting or "qwen2p5_7b_instruct_python_tool_10turn",
            "--molt-path", str(tmp_path / "molt checkout"), "--agent-path", str(tmp_path / "agent.py"),
            "--train-data", str(tmp_path / "train"), "--eval-data", str(tmp_path / "eval"),
            "--output", str(tmp_path / "output"), "--actor-gpus", "4", "--rollout-engines", "4"]


@pytest.fixture(scope="module")
def native_cli():
    """Execute the pinned CLI's actual argparse and validation code without loading GPUs.

    Common argument helpers are stdlib-only. Only the training call and imports
    outside the CLI's __main__ block are excluded; validation isn't reimplemented.
    """
    source = os.environ.get("MOLT_SOURCE_PATH")
    if source is None:
        pytest.skip("Set MOLT_SOURCE_PATH to check commands against the pinned backend")
    root = Path(source)
    EXPERIMENT.verify_backend(root)
    namespace = {"argparse": argparse}
    namespace.update(runpy.run_path(str(root / "molt/cli/common_args.py")))
    namespace.update(runpy.run_path(str(root / "molt/utils/config.py")))
    experience = ast.parse((root / "molt/trainer/algorithm/experience.py").read_text())
    size_fn = next(n for n in experience.body if isinstance(n, ast.FunctionDef)
                   and n.name == "get_model_parallel_size")
    exec(compile(ast.Module(body=[size_fn], type_ignores=[]), "native_parallel_size", "exec"), namespace)
    tree = ast.parse((root / "molt/cli/train_rl_ray.py").read_text())
    entry = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")
    body = [n for n in entry.body if not (isinstance(n, ast.ImportFrom)
            and n.module in ("molt.cli.common_args", "molt.utils.config"))]
    code = compile(ast.Module(body=body, type_ignores=[]), "native_train_cli", "exec")
    namespace["train"] = lambda args: None
    return code, namespace


@pytest.mark.parametrize("setting", EXPERIMENT.SETTING_NAMES)
@pytest.mark.parametrize("route", ["token", "sequence"])
@pytest.mark.parametrize("estimator", ["mc", "topk", "binary", "full"])
def test_all_experiment_commands_pass_native_validation(setting, route, estimator, native_cli,
                                                       tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_experiment", *launch_arguments(tmp_path, setting),
                                     "--route", route, "--kl-estimator", estimator, "--dry-run"])
    EXPERIMENT.main()
    command = shlex.split(capsys.readouterr().out.splitlines()[2])
    native_args = command[command.index("--") + 1:]
    monkeypatch.setattr(sys, "argv", ["train_rl_ray", *native_args])
    code, namespace = native_cli
    exec(code, namespace)
    args = namespace["args"]
    assert args.actor.klpo_route == route and args.actor.klpo_kl_estimator == estimator
    assert args.rollout.batch_size == args.train.batch_size == args.rollout.vllm_generate_batch_size
    assert args.rollout.n_samples_per_prompt == 1
    assert args.train.force_sync_mode and args.train.async_queue_size == 1
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("setting", EXPERIMENT.SETTING_NAMES)
def test_only_single_turn_math_defaults_to_builtin_agent(setting, tmp_path, monkeypatch, capsys):
    arguments = launch_arguments(tmp_path, setting)
    index = arguments.index("--agent-path")
    del arguments[index:index + 2]
    monkeypatch.setattr(sys, "argv", ["run_experiment", *arguments, "--dry-run"])
    if EXPERIMENT.load_settings(setting)["agent_kind"] != "math":
        with pytest.raises(SystemExit):
            EXPERIMENT.main()
        assert "--agent-path is required" in capsys.readouterr().err
        return
    EXPERIMENT.main()
    command = shlex.split(capsys.readouterr().out.splitlines()[2])
    assert command[command.index("--max-agent-turns") + 1] == "1"
    assert command[command.index("--train.agent_path") + 1] == str(tmp_path / "molt checkout/examples/python/agents/math.py")


@pytest.mark.parametrize("extra, message", [
    (["--route", "sequence", "--mc-samples", "1"], "M >= 2"),
    (["--beta", "nan"], "positive and finite"),
    (["--top-k", "0"], "K/M must be positive"),
    (["--episodes", "0"], "episodes must be positive"),
])
def test_invalid_experiments_fail_even_in_preview(extra, message):
    result = subprocess.run([sys.executable, str(ROOT / "examples/run_experiment.py"),
                             "--setting", EXPERIMENT.SETTING_NAMES[0], "--print-config", *extra],
                            capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr


def test_default_token_mc_allows_one_auxiliary_draw():
    result = subprocess.run([sys.executable, str(ROOT / "examples/run_experiment.py"),
                             "--setting", EXPERIMENT.SETTING_NAMES[0], "--print-config", "--mc-samples", "1"],
                            capture_output=True, text=True, check=True)
    flags = json.loads(result.stdout)["flags"]
    assert (flags["actor.klpo_route"], flags["actor.klpo_kl_estimator"], flags["actor.klpo_mc_samples"]) == ("token", "mc", 1)


@pytest.mark.parametrize("missing", [None, "agent", "alfworld_data", "capability"])
def test_launch_forwards_environment_and_records_settings(tmp_path, monkeypatch, capsys, missing):
    arguments = launch_arguments(tmp_path, "qwen2p5_7b_instruct_alfworld_50turn")
    monkeypatch.setattr(sys, "argv", ["run_experiment", *arguments, "--dry-run"])
    EXPERIMENT.main()
    command = shlex.split(capsys.readouterr().out.splitlines()[2])
    supported = [arg for arg in command[command.index("--") + 1:] if arg.startswith("--")]
    if missing == "capability":
        supported.remove("--actor.klpo_mc_samples")
    backend = tmp_path / "molt checkout"
    cli = backend / "molt/cli"
    cli.mkdir(parents=True)
    for path in (backend / "molt/__init__.py", cli / "__init__.py"):
        path.touch()
    (cli / "train_rl_ray.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "def _ray_runtime_env_vars(): return {'NCCL_DEBUG': os.environ['NCCL_DEBUG']}\n"
        "if __name__ == '__main__':\n"
        f"    if '--help' in sys.argv: print({' '.join(supported)!r})\n"
        "    else: Path('executed.json').write_text(json.dumps({'argv': sys.argv, 'turns': os.environ['MAX_AGENT_TURNS']}))\n"
    )
    (backend / "ray.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def init(**kwargs): Path('runtime.json').write_text(json.dumps(kwargs))\n"
    )
    for name in ("train", "eval", "games"):
        (tmp_path / name).mkdir()
    if missing != "agent":
        (tmp_path / "agent.py").touch()
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    if missing != "alfworld_data":
        monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path / "games"))
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    # Fake only the source revision check; both launcher subprocesses execute.
    monkeypatch.setattr(EXPERIMENT, "verify_backend", lambda root: None)
    monkeypatch.setattr(sys, "argv", ["run_experiment", *arguments])
    if missing:
        with pytest.raises(SystemExit):
            EXPERIMENT.main()
        assert not (tmp_path / "output").exists()
        assert not (backend / "executed.json").exists()
        return
    EXPERIMENT.main()
    config = json.loads((tmp_path / "output/resolved_settings.json").read_text())
    execution = json.loads((backend / "executed.json").read_text())
    runtime = json.loads((backend / "runtime.json").read_text())
    assert config["molt_revision"] == EXPERIMENT.MOLT_REVISION
    assert execution["turns"] == "50"
    assert runtime["runtime_env"]["env_vars"] == {"NCCL_DEBUG": "INFO", **config["worker_env"]}
    assert execution["argv"][1:] == command[command.index("--") + 1:]


@pytest.mark.parametrize("route", ["token", "sequence"])
@pytest.mark.parametrize("estimator", ["mc", "topk", "binary", "full"])
def test_tool_observations_do_not_change_trajectory_loss_or_gradients(route, estimator):
    torch.manual_seed(42)
    logits = torch.randn(1, 8, 5, dtype=torch.float64, requires_grad=True)
    p = logits.log_softmax(-1)
    q = torch.randn_like(logits).log_softmax(-1)
    # Prompt, assistant turn 1, tool output, assistant turn 2, trailing padding.
    mask = torch.tensor([[False, True, True, False, False, True, True, False]])
    actions = torch.tensor([[0, 1, 2, 3, 4, 1, 0, 2]])
    logp = p.gather(-1, actions[..., None]).squeeze(-1)
    logq = q.gather(-1, actions[..., None]).squeeze(-1)
    paux = qaux = None
    if estimator != "binary":
        ids = q.topk(2, -1).indices if estimator == "topk" else torch.tensor([0, 0, 3]).expand(1, 8, 3)
        paux, qaux = (p, q) if estimator == "full" else (p.gather(-1, ids), q.gather(-1, ids))
    adapter = KLPOLoss(route=route, kl_estimator=estimator, beta=.2)

    def loss(compact):
        positions = mask[0] if compact else torch.ones(8, dtype=torch.bool)
        return adapter(logp[:, positions], None, None, action_mask=mask[:, positions],
                       rollout_log_probs=logq[:, positions], rewards=torch.tensor([1.]), global_batch_size=1,
                       kl_log_probs=paux[:, positions] if paux is not None else None,
                       behavior_kl_log_probs=qaux[:, positions] if qaux is not None else None,
                       full_vocabulary=estimator == "full")[0]

    complete, compact = loss(False), loss(True)
    torch.testing.assert_close(complete, compact)
    grad, = torch.autograd.grad(complete, logits, retain_graph=True)
    expected, = torch.autograd.grad(compact, logits)
    torch.testing.assert_close(grad, expected)
    assert torch.count_nonzero(grad[~mask]) == 0
    assert torch.count_nonzero(grad[mask]) > 0
