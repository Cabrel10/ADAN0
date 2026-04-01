"""
Root conftest.py – loads .env variables before any test uses ConfigLoader.
"""
import os
from pathlib import Path


def _load_dotenv():
    """Load .env file from project root into os.environ (no external deps)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


# Auto-execute on import (pytest collects conftest before tests)
_load_dotenv()
