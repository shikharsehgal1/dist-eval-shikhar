"""conftest.py – project root on sys.path so every test can ``import disteval``."""
import sys
import os

# Insert the project root (directory containing disteval/) at the front of the
# module search path.  This makes ``import disteval`` work from any test file
# regardless of how pytest is invoked.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
