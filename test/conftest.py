"""Pytest configuration file for hybridlane tests."""

import sys
from pathlib import Path

# Add the src directory to the Python path to ensure imports work
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))