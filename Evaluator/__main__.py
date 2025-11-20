"""Module entrypoint for interactive LM Studio evaluations."""
from __future__ import annotations

import os
import sys

# Allow invocation as `python Evaluator` (directory as script) by ensuring the
# repo root is on sys.path before importing the package module.
if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from Evaluator.interactive_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
