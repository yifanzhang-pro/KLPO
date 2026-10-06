"""Verify the native KLPO backend checkout without modifying any source files."""

import argparse
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MOLT_REPOSITORY = "https://github.com/yifanzhang-pro/labs-molt.git"
MOLT_SUBMODULE = ROOT / "external" / "labs-molt"
KLPO_API_VERSION = 1


def pinned_revision():
    """Return the labs-molt commit recorded by the external/labs-molt submodule."""
    entry = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--stage", "--", "external/labs-molt"],
                                    text=True).split()
    if len(entry) < 2 or entry[0] != "160000":
        raise RuntimeError("external/labs-molt is not a submodule of this KLPO checkout")
    return entry[1]


MOLT_REVISION = pinned_revision()


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
    parser.add_argument("--molt-path", type=Path, default=MOLT_SUBMODULE,
                        help="labs-molt checkout (default: the external/labs-molt submodule)")
    args = parser.parse_args()
    verify_backend(args.molt_path.expanduser().resolve())
    print(f"Native KLPO backend verified at {MOLT_REVISION}")


if __name__ == "__main__":
    main()
