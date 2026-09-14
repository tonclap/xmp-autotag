"""Runtime configuration, read from the environment (and from an optional .env).

Everything machine-specific lives here: where the archive is, which model to
call, where the embedding model was unpacked. No other module hardcodes a path,
so the same checkout works on Windows, macOS and Linux.

Values are resolved once at import time. Real environment variables win over
.env, so a one-off run can override a setting without editing files:

    PHOTO_ROOT=/mnt/photos python scripts/tag_archive.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Minimal .env reader (no dependency on python-dotenv).

    Uses setdefault: an existing environment variable is never overwritten.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")


def _str(name, default=""):
    return os.environ.get(name, "").strip() or default


def _path(name, default=None):
    raw = _str(name)
    return Path(raw).expanduser() if raw else default


def _int(name, default):
    raw = _str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"{name} must be an integer, got {raw!r}")


def _list(name):
    return [part.strip() for part in _str(name).split(",") if part.strip()]


# --- Archive layout ------------------------------------------------------
# PHOTO_ROOT is the personal photo archive: scanned recursively.
# MEDIA_ROOT is an optional second collection with different content
# (screenshots, infographics, memes, reference images). It is scanned only
# inside MEDIA_SUBDIRS, because such a folder usually also holds things that
# make no sense to describe (fonts, PDFs, course material).
PHOTO_ROOT = _path("PHOTO_ROOT")
MEDIA_ROOT = _path("MEDIA_ROOT")
MEDIA_SUBDIRS = _list("MEDIA_SUBDIRS")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".orf"}

# Run artefacts: JSONL logs, sidecar backups, the search index, thumbnails.
# Namespaced deliberately: a bare OUTPUT_DIR already exists in the environment of
# many machines, and inheriting it silently sends this tool's index and backups
# into some other tool's folder (caught exactly that way during a smoke test).
OUTPUT_DIR = _path("XMP_AUTOTAG_OUTPUT_DIR", REPO_ROOT / "output")

# --- Vision model (OpenRouter) -------------------------------------------
API_KEY = _str("OPENROUTER_API_KEY")
MODEL = _str("OPENROUTER_MODEL", "qwen/qwen3.7-flash")
# Language the model must write descriptions and keywords in. It goes into the
# prompt verbatim, so write it the way you would say it to a person: "English",
# "German", "Brazilian Portuguese".
TAG_LANGUAGE = _str("TAG_LANGUAGE", "English")
MAX_WORKERS = _int("XMP_AUTOTAG_WORKERS", 8)  # namespaced for the same reason

# --- Semantic search -----------------------------------------------------
# Directory with an ONNX text embedding model: model_quantized.onnx +
# tokenizer.json. See README, "Semantic search" — nothing is downloaded
# automatically.
EMBEDDING_MODEL_DIR = _path("EMBEDDING_MODEL_DIR", REPO_ROOT / "models" / "embedding")
WEB_SEARCH_PORT = _int("WEB_SEARCH_PORT", 8766)
# Optional JSON file mapping a geocoded name to query aliases in your own
# language, e.g. {"austria": ["osterreich"]}. See scripts/search_core.py.
GEO_ALIASES_FILE = _path("GEO_ALIASES_FILE")


def require_photo_root():
    """PHOTO_ROOT, or exit with an actionable message instead of a traceback."""
    if PHOTO_ROOT is None:
        sys.exit(
            "PHOTO_ROOT is not set. Copy .env.example to .env and point "
            "PHOTO_ROOT at your photo archive."
        )
    if not PHOTO_ROOT.is_dir():
        sys.exit(f"PHOTO_ROOT does not exist or is not a directory: {PHOTO_ROOT}")
    return PHOTO_ROOT


def require_api_key():
    """The OpenRouter key, checked only when a request is about to be made.

    Search, indexing and duplicate detection never call this: browsing an
    already tagged archive must work without a key.
    """
    if not API_KEY or API_KEY.endswith("..."):
        sys.exit(
            "OPENROUTER_API_KEY is not set (see .env.example). It is needed "
            "only for tagging; search and indexing run fully offline."
        )
    return API_KEY


def output_path(*parts):
    """Path inside OUTPUT_DIR, creating the directory on first use."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR.joinpath(*parts)
