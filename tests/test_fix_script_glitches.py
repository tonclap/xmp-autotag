"""Tests for scripts/fix_script_glitches.py: the repair pass for wrong-script text.

The interesting property is what the pass refuses to do: a re-tag that comes
back glitched again must leave the sidecar exactly as it was, because writing
one broken description over another helps nobody. No network — the model call is
replaced with a canned answer.
"""
import json

import pytest

import fix_script_glitches as fix
import tag_archive
import xmp


def _index(tmp_path, records):
    path = tmp_path / "search_index.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n{truncated", encoding="utf-8")
    return path


def test_affected_paths_finds_only_the_glitched_records(tmp_path):
    index_path = _index(tmp_path, [
        {"path": "/a.jpg", "description": "A clean description", "keywords": ["dog"]},
        {"path": "/b.jpg", "description": "A description with 猫 in it", "keywords": ["cat"]},
        {"path": "/c.jpg", "description": "Clean here", "keywords": ["猫", "yard"]},
    ])

    # The truncated last line is what an interrupted indexing run leaves behind.
    assert fix.affected_paths(index_path) == ["/b.jpg", "/c.jpg"]


def test_a_missing_index_exits_with_an_actionable_message(tmp_path):
    with pytest.raises(SystemExit) as exc:
        fix.affected_paths(tmp_path / "nope.jsonl")
    assert "build_search_index.py" in str(exc.value)


def test_already_attempted_reads_the_log_and_tolerates_junk(tmp_path):
    log_path = tmp_path / "fix.jsonl"
    log_path.write_text(
        '{"path": "/a.jpg", "status": "fixed"}\n'
        "\n"
        '{"status": "fixed"}\n'          # no path key
        "not json\n"
        '{"path": "/b.jpg", "status": "still_glitched"}\n',
        encoding="utf-8",
    )

    # "attempted", not "clean": a still_glitched file counts as done for this
    # run, which is why the module docstring tells you to delete log lines.
    assert fix.already_attempted(log_path) == {"/a.jpg", "/b.jpg"}


def test_no_log_yet_means_nothing_attempted(tmp_path):
    assert fix.already_attempted(tmp_path / "nope.jsonl") == set()


@pytest.fixture
def fix_run(monkeypatch, tmp_path):
    """A FixRun over a tagged sidecar, with the model call stubbed out."""
    source = tag_archive.Source(
        name="photos", root=tmp_path, prompt=lambda ctx: "prompt", reasoning=True
    )
    run = fix.FixRun(source, reasoning=True, log_path=tmp_path / "fix.jsonl")

    img = tmp_path / "a.jpg"
    img.write_bytes(b"")
    xmp_path = tmp_path / "a.xmp"
    xmp.create_new(xmp_path, *xmp.build_fields("Описание с 猫 внутри", ["猫", "двор"]))

    def install(answer):
        monkeypatch.setattr(
            tag_archive.vlm, "describe", lambda *a, **kw: answer
        )

    return run, install, img, xmp_path


def test_a_clean_answer_replaces_both_fields(fix_run):
    run, install, img, xmp_path = fix_run
    install({"text": "Кот во дворе.\n\nкот, двор", "cost": 0.0001})

    run.process_one(img)

    content = xmp_path.read_text(encoding="utf-8")
    assert "猫" not in content
    assert "Кот во дворе." in content
    assert run.stats["fixed"] == 1


def test_an_answer_that_is_still_glitched_leaves_the_file_alone(fix_run):
    run, install, img, xmp_path = fix_run
    before = xmp_path.read_text(encoding="utf-8")
    install({"text": "Кот во дворе 猫.\n\nкот, двор", "cost": 0.0001})

    run.process_one(img)

    assert xmp_path.read_text(encoding="utf-8") == before
    assert run.stats["still_glitched"] == 1
    logged = json.loads((run.log_path).read_text(encoding="utf-8").splitlines()[0])
    assert logged["status"] == "still_glitched"


def test_an_unparsable_answer_leaves_the_file_alone(fix_run):
    run, install, img, xmp_path = fix_run
    before = xmp_path.read_text(encoding="utf-8")
    install({"text": "Just prose, and nothing that looks like a keyword list."})

    run.process_one(img)

    assert xmp_path.read_text(encoding="utf-8") == before
    assert run.stats["error"] == 1


def test_a_null_content_is_recorded_as_an_error(fix_run):
    run, install, img, _xmp_path = fix_run
    install({"text": None})

    run.process_one(img)

    assert run.stats["error"] == 1
    logged = json.loads((run.log_path).read_text(encoding="utf-8").splitlines()[0])
    assert "null content" in logged["error"]
