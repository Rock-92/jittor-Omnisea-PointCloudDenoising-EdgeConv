#!/usr/bin/env python
"""Evaluation entry point.

All arguments are forwarded to the repository-level evaluate.py script.
"""

from pathlib import Path
import os
import runpy
import sys


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    sys.argv = [str(repo_root / "evaluate.py"), *sys.argv[1:]]
    runpy.run_path(str(repo_root / "evaluate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
