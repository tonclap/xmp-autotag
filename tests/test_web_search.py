"""Tests for scripts/web_search.py: the path guard in front of the disk.

The UI serves images straight off the filesystem and has no authentication, so
safe_photo_path is the only thing standing between a query string and any file
on the machine. Everything here runs on tmp_path; no server is started and no
thumbnail is rendered.
"""
import pytest

import web_search


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A photo root with one image, and a secret outside it."""
    root = tmp_path / "photos"
    (root / "2019").mkdir(parents=True)
    photo = root / "2019" / "a.jpg"
    photo.write_bytes(b"jpeg")
    (tmp_path / "secret.txt").write_text("not yours", encoding="utf-8")
    monkeypatch.setattr(web_search, "ALLOWED_ROOTS", [root])
    return root, photo


def test_a_path_inside_the_root_is_served(archive):
    _root, photo = archive
    assert web_search.safe_photo_path(str(photo)) == photo.resolve()


def test_a_path_outside_the_root_is_refused(archive, tmp_path):
    assert web_search.safe_photo_path(str(tmp_path / "secret.txt")) is None


def test_traversal_out_of_the_root_is_refused(archive):
    root, _photo = archive
    assert web_search.safe_photo_path(f"{root}/2019/../../secret.txt") is None


def test_a_directory_is_not_an_image(archive):
    root, _photo = archive
    assert web_search.safe_photo_path(str(root / "2019")) is None


def test_a_missing_file_inside_the_root_is_refused(archive):
    root, _photo = archive
    assert web_search.safe_photo_path(str(root / "2019" / "gone.jpg")) is None


def test_an_empty_or_absent_parameter_is_refused(archive):
    assert web_search.safe_photo_path("") is None
    assert web_search.safe_photo_path(None) is None


def test_a_second_root_is_also_allowed(tmp_path, monkeypatch):
    # PHOTO_ROOT and MEDIA_ROOT are separate trees; a hit on either is fine,
    # and a miss on the first must not end the search.
    photos = tmp_path / "photos"
    media = tmp_path / "media"
    photos.mkdir()
    media.mkdir()
    meme = media / "b.png"
    meme.write_bytes(b"png")
    monkeypatch.setattr(web_search, "ALLOWED_ROOTS", [photos, media])

    assert web_search.safe_photo_path(str(meme)) == meme.resolve()


def test_no_roots_configured_serves_nothing(tmp_path, monkeypatch):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"jpeg")
    monkeypatch.setattr(web_search, "ALLOWED_ROOTS", [])
    assert web_search.safe_photo_path(str(photo)) is None
