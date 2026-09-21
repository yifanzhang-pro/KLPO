"""Verify the native KLPO backend checkout without modifying any source files."""

import argparse
import ast
from pathlib import Path
import subprocess

MOLT_REPOSITORY = "https://github.com/yifanzhang-pro/labs-molt.git"
MOLT_REVISION = "e24e22faaf0d3cf3ad56dec47215117d95914be2"
KLPO_API_VERSION = 2


def verify_backend(root: Path):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != MOLT_REVISION:
        raise ValueError(f"Expected labs-molt {MOLT_REVISION}; found {revision}")
    module = ast.parse((root / "molt/__init__.py").read_text())
    versions = [ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "KLPO_API_VERSION" for t in node.targets)]
    if versions != [KLPO_API_VERSION]:
        raise ValueError("Backend does not expose the required native KLPO API")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError("Native backend has tracked modifications; use the pinned clean checkout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molt-path", required=True, type=Path)
    args = parser.parse_args()
    verify_backend(args.molt_path.expanduser().resolve())
    print(f"Native KLPO backend verified at {MOLT_REVISION}")


if __name__ == "__main__":
    main()
