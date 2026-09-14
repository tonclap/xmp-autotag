"""Tests for scripts/config.py: reading settings without surprising anyone.

The two behaviours worth pinning: a real environment variable always wins over
.env, and a missing setting produces a readable message instead of a traceback
somewhere deep in a thread pool.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import config

SCRIPTS_DIR = Path(config.__file__).resolve().parent


def _run_snippet(body, env_file=None, env=None, cwd=None):
    """Import config in a fresh interpreter, since it resolves values once."""
    script = textwrap.dedent(body)
    full_env = {"PATH": "", "SYSTEMROOT": "C:\\Windows"}
    full_env.update(env or {})
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**{k: v for k, v in full_env.items() if v is not None}},
        cwd=cwd,
    )
    return result


def test_dotenv_fills_gaps_but_never_overrides_the_environment(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "config.py").write_bytes((SCRIPTS_DIR / "config.py").read_bytes())
    (repo / ".env").write_text(
        "OPENROUTER_MODEL=from/dotenv\nTAG_LANGUAGE=German\n", encoding="utf-8"
    )

    result = _run_snippet(
        """
        import sys
        sys.path.insert(0, "scripts")
        import config
        print(config.MODEL)
        print(config.TAG_LANGUAGE)
        """,
        env={"TAG_LANGUAGE": "Spanish"},
        cwd=repo,
    )

    assert result.returncode == 0, result.stderr
    model, language = result.stdout.split()
    assert model == "from/dotenv"  # taken from .env
    assert language == "Spanish"  # the real environment wins


def test_missing_photo_root_exits_with_an_actionable_message(monkeypatch):
    monkeypatch.setattr(config, "PHOTO_ROOT", None)
    with pytest.raises(SystemExit) as exc:
        config.require_photo_root()
    assert "PHOTO_ROOT" in str(exc.value) and ".env" in str(exc.value)


def test_photo_root_pointing_at_nothing_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PHOTO_ROOT", tmp_path / "nope")
    with pytest.raises(SystemExit) as exc:
        config.require_photo_root()
    assert "does not exist" in str(exc.value)


def test_missing_api_key_mentions_that_search_does_not_need_one(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    with pytest.raises(SystemExit) as exc:
        config.require_api_key()
    assert "offline" in str(exc.value)


def test_placeholder_api_key_is_treated_as_unset(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "sk-or-...")
    with pytest.raises(SystemExit):
        config.require_api_key()


def test_output_path_creates_the_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    path = config.output_path("run.jsonl")
    assert path == tmp_path / "out" / "run.jsonl"
    assert path.parent.is_dir()


def test_non_numeric_worker_count_fails_loudly(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "config.py").write_bytes((SCRIPTS_DIR / "config.py").read_bytes())

    result = _run_snippet(
        """
        import sys
        sys.path.insert(0, "scripts")
        import config
        """,
        env={"XMP_AUTOTAG_WORKERS": "lots"},
        cwd=repo,
    )

    assert result.returncode != 0
    assert "XMP_AUTOTAG_WORKERS must be an integer" in result.stderr
