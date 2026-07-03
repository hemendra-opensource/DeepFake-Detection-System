"""
conftest.py
============
Pytest configuration — adds the project root to sys.path so that all
test modules can import project packages without needing installation.
"""

import sys
from pathlib import Path

# Insert the project root at the front of sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
