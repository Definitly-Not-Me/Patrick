#!/usr/bin/env python3
"""Local training launcher.

Thin wrapper: train.py owns the full CLI (--test/--resume) and the Kaggle
environment guards. Kaggle notebooks embed train.py directly and call its
main() themselves, so keep this dual entry in sync with train.py's main().
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import main

if __name__ == "__main__":
    main()