"""
Puts src/ on sys.path so tests can import sibling modules the same way
they always have (`from load_dataset import ...`, `from baseline.baseline1
import ...`) even though this test/ package lives one directory below
them, not alongside them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
