"""Tests for scripts/tag_archive.py: discovery and the idempotency check.

No real archive and no API call: discover() walks a synthetic tree in tmp_path
and already_done() reads synthetic sidecars. These two functions decide what a
run costs money for, so they are worth pinning.
"""
import tag_archive
import xmp


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_discover_filters_by_extension_and_recurses(tmp_path):
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "b.JPEG")  # extension match is case-insensitive
    _touch(tmp_path / "c.txt")  # not an image
    _touch(tmp_path / "d.xmp")  # a sidecar, not an image
    _touch(tmp_path / "2020" / "holiday" / "e.orf")
    _touch(tmp_path / "2020" / "holiday" / "f.heic")

    assert {p.name for p in tag_archive.discover(tmp_path)} == {
        "a.jpg", "b.JPEG", "e.orf", "f.heic"
    }


def test_discover_empty_root(tmp_path):
    assert list(tag_archive.discover(tmp_path)) == []


def test_discover_missing_root_is_not_an_error(tmp_path):
    assert list(tag_archive.discover(tmp_path / "nope")) == []


def test_discover_with_subdirs_walks_only_those(tmp_path):
    _touch(tmp_path / "charts" / "a.png")
    _touch(tmp_path / "memes" / "deep" / "b.jpg")
    _touch(tmp_path / "fonts" / "c.png")  # outside the listed subdirs
    _touch(tmp_path / "d.jpg")  # directly in the root, also outside

    found = {p.name for p in tag_archive.discover(tmp_path, subdirs=("charts", "memes"))}
    assert found == {"a.png", "b.jpg"}


def test_discover_skips_a_missing_subdir(tmp_path):
    _touch(tmp_path / "charts" / "a.png")
    found = {p.name for p in tag_archive.discover(tmp_path, subdirs=("charts", "gone"))}
    assert found == {"a.png"}


def test_discover_honours_a_custom_extension_set(tmp_path):
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "b.png")
    assert {p.name for p in tag_archive.discover(tmp_path, exts={".png"})} == {"b.png"}


def test_already_done_reads_dc_subject(tmp_path):
    tagged = tmp_path / "photo.xmp"
    tagged.write_text("<a><dc:subject>x</dc:subject></a>", encoding="utf-8")
    assert tag_archive.already_done(tagged) is True

    untagged = tmp_path / "other.xmp"
    untagged.write_text("<a><MY:Rating>5</MY:Rating></a>", encoding="utf-8")
    assert tag_archive.already_done(untagged) is False

    assert tag_archive.already_done(tmp_path / "missing.xmp") is False


def test_source_paths_are_derived_from_the_name(tmp_path, monkeypatch):
    monkeypatch.setattr(tag_archive.config, "OUTPUT_DIR", tmp_path)
    source = tag_archive.Source(
        name="photos", root=tmp_path, prompt=lambda ctx: ctx, reasoning=False
    )
    assert source.log_path == tmp_path / "tag_photos_log.jsonl"
    assert source.backup_dir == tmp_path / "xmp_backup_photos"


def test_run_writes_one_json_line_per_logged_event(tmp_path, monkeypatch):
    monkeypatch.setattr(tag_archive.config, "OUTPUT_DIR", tmp_path)
    source = tag_archive.Source(
        name="photos", root=tmp_path, prompt=lambda ctx: ctx, reasoning=False
    )
    run = tag_archive.Run(source, log_path=tmp_path / "run.jsonl")
    run.log({"path": "a.jpg", "status": "created"})
    run.log({"path": "b.jpg", "status": "error", "error": "boom"})
    run.bump("created")
    run.bump("error", cost=0.25)

    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert run.stats["created"] == 1
    assert run.stats["error"] == 1
    assert run.stats["cost"] == 0.25


def test_xmp_backup_dir_is_redirectable(tmp_path):
    # tag_archive points xmp.BACKUP_DIR at the run's own folder; nothing else
    # should be able to write outside it.
    original = xmp.BACKUP_DIR
    try:
        xmp.BACKUP_DIR = tmp_path / "backups"
        sidecar = tmp_path / "a.xmp"
        sidecar.write_text(
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            ' <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            " </rdf:Description></rdf:RDF>",
            encoding="utf-8",
        )
        backup_path, err = xmp.merge_existing(sidecar, *xmp.build_fields("Scene", ["kw"]))
        assert err is None
        assert backup_path.parent == tmp_path / "backups"
    finally:
        xmp.BACKUP_DIR = original
