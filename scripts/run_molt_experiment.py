"""Run Molt experiments with agent settings propagated to its Ray workers.

Called by examples/run_experiment.py from the verified Molt checkout. This
initializes the driver's Ray connection, without starting/stopping a shared
cluster or patching Molt. The native CLI otherwise omits these agent variables.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-agent-turns", type=int, required=True)
    parser.add_argument("molt_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.max_agent_turns < 1:
        parser.error("max-agent-turns must be positive")
    os.environ["MAX_AGENT_TURNS"] = str(args.max_agent_turns)
    sys.path.insert(0, str(Path.cwd()))
    import ray
    from molt.cli import train_rl_ray

    worker_env = train_rl_ray._ray_runtime_env_vars()
    worker_env["MAX_AGENT_TURNS"] = os.environ["MAX_AGENT_TURNS"]
    if os.environ.get("ALFWORLD_DATA"):
        worker_env["ALFWORLD_DATA"] = os.environ["ALFWORLD_DATA"]
    ray.init(runtime_env={"env_vars": worker_env})
    arguments = args.molt_args[1:] if args.molt_args[:1] == ["--"] else args.molt_args
    sys.argv = [train_rl_ray.__file__, *arguments]
    runpy.run_path(train_rl_ray.__file__, run_name="__main__")


if __name__ == "__main__":
    main()
