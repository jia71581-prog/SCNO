"""Activate only the native400 release's copied compatibility dependencies."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
def activate():
    for name in ('shared', 'r21', 'r22', 'r24', 'r25'):
        path = str(ROOT / name)
        if path not in sys.path:
            sys.path.insert(0, path)
    return ROOT
