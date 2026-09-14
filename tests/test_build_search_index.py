"""Tests for scripts/build_search_index.py: pulling facts out of a sidecar.

Synthetic XMP strings only — no archive is read and no model is loaded (the
embedder is imported but never called, since it loads lazily).
"""
from pathlib import Path

import build_search_index as bsi


# --- extract_date: EXIF > file name > folder name ------------------------

def test_exif_date_wins_over_filename_and_folder():
    content = '<x xmp:CreateDate="2020-05-12T10:00:00"/>'
    root = Path("/archive")
    path = root / "2019, 1 January" / "IMG_20180101_000000.jpg"
    assert bsi.extract_date(content, path, root) == {
        "year": 2020, "month": 5, "day": 12, "source": "exif"
    }


def test_out_of_range_exif_falls_back_to_filename():
    content = '<x exif:DateTimeOriginal="1699-01-01"/>'
    root = Path("/archive")
    path = root / "misc" / "IMG_20150407_123456.jpg"
    assert bsi.extract_date(content, path, root) == {
        "year": 2015, "month": 4, "day": 7, "source": "filename"
    }


def test_folder_name_is_the_last_resort():
    root = Path("/archive")
    path = root / "2017, 4-8 May. Trip to the coast" / "photo.jpg"
    assert bsi.extract_date("<x></x>", path, root) == {
        "year": 2017, "month": 5, "day": 4, "source": "folder"
    }


def test_no_date_signal_anywhere():
    root = Path("/archive")
    assert bsi.extract_date("<x></x>", root / "misc" / "photo.jpg", root) is None


def test_path_outside_the_root_still_tries_exif_and_filename():
    # Defensive: a stale index entry may point outside the configured root.
    content = '<x xmp:CreateDate="2021-07-04T00:00:00"/>'
    assert bsi.extract_date(content, Path("/elsewhere/a.jpg"), Path("/archive")) == {
        "year": 2021, "month": 7, "day": 4, "source": "exif"
    }
    assert bsi.extract_date("<x></x>", Path("/elsewhere/a.jpg"), Path("/archive")) is None


# --- extract_people ------------------------------------------------------

def test_extract_people_parses_and_trims_names():
    content = (
        "<Iptc4xmpExt:PersonInImage><rdf:Bag>"
        "<rdf:li>Anna</rdf:li><rdf:li>  Boris  </rdf:li><rdf:li>   </rdf:li>"
        "</rdf:Bag></Iptc4xmpExt:PersonInImage>"
    )
    assert bsi.extract_people(content) == ["Anna", "Boris"]


def test_extract_people_absent_tag():
    assert bsi.extract_people("<x></x>") == []


# --- extract_location ----------------------------------------------------

def test_extract_location_full_fields():
    content = (
        "<Iptc4xmpExt:LocationShown><rdf:Bag><rdf:li "
        'Iptc4xmpExt:CountryName="Austria" Iptc4xmpExt:City="Vienna" '
        'Iptc4xmpExt:ProvinceState="Vienna">'
        "<Iptc4xmpExt:LocationName><rdf:Alt><rdf:li>Schonbrunn</rdf:li></rdf:Alt>"
        "</Iptc4xmpExt:LocationName>"
        "</rdf:li></rdf:Bag></Iptc4xmpExt:LocationShown>"
    )
    assert bsi.extract_location(content) == {
        "country": "Austria", "city": "Vienna", "province": "Vienna", "name": "Schonbrunn"
    }


def test_extract_location_all_attrs_empty_is_treated_as_absent():
    content = (
        '<Iptc4xmpExt:LocationShown><rdf:Bag><rdf:li Iptc4xmpExt:CountryName="">'
        "</rdf:li></rdf:Bag></Iptc4xmpExt:LocationShown>"
    )
    assert bsi.extract_location(content) is None
    assert bsi.extract_location("<x></x>") is None


# --- extract: the whole sidecar -----------------------------------------

def test_extract_full_pipeline(tmp_path):
    content = (
        '<rdf:Description rdf:about="" xmp:CreateDate="2021-03-15T00:00:00">\n'
        "<dc:subject><rdf:Bag><rdf:li>cat</rdf:li><rdf:li>yard</rdf:li></rdf:Bag></dc:subject>\n"
        '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">A cat in the yard</rdf:li>'
        "</rdf:Alt></dc:description>\n"
        "<Iptc4xmpExt:PersonInImage><rdf:Bag><rdf:li>Anna</rdf:li></rdf:Bag>"
        "</Iptc4xmpExt:PersonInImage>\n"
        "</rdf:Description>"
    )
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    description, keywords, people, location, date = bsi.extract(
        xmp_path, tmp_path / "photo.jpg", tmp_path
    )

    assert description == "A cat in the yard"
    assert keywords == ["cat", "yard"]
    assert people == ["Anna"]
    assert location is None
    assert date == {"year": 2021, "month": 3, "day": 15, "source": "exif"}


def test_extract_without_description_returns_empty_tuple(tmp_path):
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.write_text("<rdf:Description rdf:about=''></rdf:Description>", encoding="utf-8")
    assert bsi.extract(xmp_path, tmp_path / "photo.jpg", tmp_path) == (None, [], [], None, None)


# --- already_indexed: resuming an interrupted run ------------------------

def test_already_indexed_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "INDEX_PATH", tmp_path / "nope.jsonl")
    assert bsi.already_indexed() == set()


def test_already_indexed_skips_malformed_lines(tmp_path, monkeypatch):
    index_path = tmp_path / "search_index.jsonl"
    index_path.write_text(
        '{"path": "a.jpg", "description": "x"}\n'
        "{not valid json\n"
        '{"description": "no path key"}\n'
        "\n"
        '{"path": "b.jpg", "description": "y"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(bsi, "INDEX_PATH", index_path)
    assert bsi.already_indexed() == {"a.jpg", "b.jpg"}
