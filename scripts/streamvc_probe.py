#!/usr/bin/env python3
"""Source-checkout entry point for the untrained neural-core probe."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.experimental.probe import main

if __name__ == '__main__':
    main()
