"""Shared pytest setup.

Modules in scripts/ import each other by bare name, the way a script run from
the repository root does, so scripts/ goes on sys.path once for the session.

The whole suite is offline and archive-free: no HTTP requests, no ONNX model, no
real photos. Everything runs on synthetic strings and tmp_path, which is why it
finishes in seconds on a fresh clone with no .env at all.
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
