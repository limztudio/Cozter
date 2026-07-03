"""Pytest import setup for running from the package checkout itself."""

from __future__ import annotations

import os
import sys


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_PARENT = os.path.dirname(PACKAGE_DIR)

if PACKAGE_PARENT in sys.path:
    sys.path.remove(PACKAGE_PARENT)
sys.path.insert(0, PACKAGE_PARENT)
