"""Tests for scripts/retry_failed.py: which files a second pass picks up.

No network: the model call is replaced with a list of canned answers, so what
is pinned here is the retry policy itself — which failures are worth repeating,
which are final, and that a file is only read out of the log once.
"""
from pathlib import Path

import pytest

import retry_failed
import tag_archive


def _log(tmp_path, records):
    path = tmp_path / "run.jsonl"
    lines = [
        '{"path": "%s", "status": "%s", "error": "%s"}' % (p, s, e)
        for p, s, e in records
    ]
    path.write_text("\n".join(lines) + "\n{not json at all", encoding="utf-8")
    return path


def test_failed_paths_are_unique_and_in_log_order(tmp_path):
    log_path = _log(tmp_path, [
        ("/a.jpg", "error", "HTTP 429"),
        ("/b.jpg", "created", ""),
        ("/a.jpg", "error", "HTTP 429"),       # same file failing twice
        ("/c.jpg", "error", "failed to parse"),
        ("/d.jpg", "skipped_has_subject", ""),
    ])

    # The trailing line is deliberately truncated, the way an interrupted run
    # leaves it; it must be skipped, not raise.
    assert retry_failed.failed_paths(log_path) == [Path("/a.jpg"), Path("/c.jpg")]


def test_a_missing_log_exits_with_an_actionable_message(tmp_path):
    with pytest.raises(SystemExit) as exc:
        retry_failed.failed_paths(tmp_path / "nope.jsonl")
    assert "run tag_archive.py first" in str(exc.value)


@pytest.fixture
def retry_run(monkeypatch, tmp_path):
    """A RetryRun whose model call returns canned answers, with no sleeping."""
    source = tag_archive.Source(
        name="photos", root=tmp_path, prompt=lambda ctx: "prompt", reasoning=False
    )
    run = retry_failed.RetryRun(source, log_path=tmp_path / "retry.jsonl")
    monkeypatch.setattr(retry_failed.time, "sleep", lambda _s: None)

    def install(answers):
        calls = []

        def fake_describe(path, prompt, reasoning=None):
            calls.append(path)
            return answers[min(len(calls) - 1, len(answers) - 1)]

        monkeypatch.setattr(retry_failed.vlm, "describe", fake_describe)
        return calls

    return run, install


def test_a_rate_limit_is_retried_until_it_passes(retry_run, tmp_path):
    run, install = retry_run
    calls = install([{"error": "HTTP 429: slow down"}, {"text": "A dog.\n\ndog, yard"}])

    result, _context = run.describe(tmp_path / "a.jpg")

    assert result == {"text": "A dog.\n\ndog, yard"}
    assert len(calls) == 2


def test_a_final_error_is_not_repeated(retry_run, tmp_path):
    # 413 and a parse failure do not get better by asking again, and every
    # repeat is paid for.
    run, install = retry_run
    calls = install([{"error": "HTTP 413: payload too large"}])

    result, _context = run.describe(tmp_path / "a.jpg")

    assert result["error"].startswith("HTTP 413")
    assert len(calls) == 1


def test_a_rate_limit_gives_up_after_the_last_attempt(retry_run, tmp_path):
    run, install = retry_run
    calls = install([{"error": "HTTP 429: slow down"}])

    result, _context = run.describe(tmp_path / "a.jpg")

    assert result["error"].startswith("HTTP 429")
    assert len(calls) == retry_failed.RETRY_ATTEMPTS


def test_a_network_exception_is_returned_not_raised(retry_run, tmp_path, monkeypatch):
    run, _install = retry_run

    def raising(*_a, **_kw):
        raise retry_failed.requests.ConnectionError("no route to host")

    monkeypatch.setattr(retry_failed.vlm, "describe", raising)

    result, _context = run.describe(tmp_path / "a.jpg")

    assert "request failed" in result["error"]
