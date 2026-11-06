#!/usr/bin/env python3
"""
Top-level entry script for AI Hand Gesture Mouse Controller.
Executes the main application loop.
"""

import os
import sys

# Ensure root directory is on the python search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import run_app

if __name__ == "__main__":
    exit_code = run_app()
    sys.exit(exit_code)

