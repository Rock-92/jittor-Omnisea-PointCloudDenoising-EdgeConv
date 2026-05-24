#!/usr/bin/env python
"""Training entry point for the point cloud denoising baseline."""

from pathlib import Path
import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description="Train the denoising baseline.")
    parser.add_argument(
        "--config",
        "--task",
        dest="config",
        default="configs/task/train_vm.yaml",
        help="Path to the task config.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    sys.argv = [
        str(repo_root / "run.py"),
        "--task",
        args.config,
        "--seed",
        str(args.seed),
    ]
    runpy.run_path(str(repo_root / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
