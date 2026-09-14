"""Tests for scripts/search_core.py: the metadata half of hybrid search.

Deterministic logic only — embeddings are never computed, because the model loads
lazily inside embed(). What is asserted here is that a name in a declined form
still matches, that a place matches through an alias, and that a date match earns
a bonus proportional to how precise it is.
"""
import json

import numpy as np

import search_core as sc


# --- normalisation -------------------------------------------------------

def test_norm_casefolds_and_folds_yo():
    assert sc._norm("Ёлка В ЛЕСУ") == "елка в лесу"
    assert sc._norm("Vienna") == "vienna"


def test_words_drops_digits_and_short_tokens():
    assert sc._words("cat, 12 and house-2017 by the river") == ["cat", "and", "house", "the", "river"]
    assert sc._words("кот, 12 и дом-2017 у реки") == ["кот", "дом", "реки"]


def test_prefix_overlap_handles_inflection():
    assert sc._prefix_overlap("anna", "anna's")
    assert sc._prefix_overlap("мария", "марии")
    assert not sc._prefix_overlap("anna", "boris")


def test_prefix_overlap_requires_exact_match_for_short_words():
    assert sc._prefix_overlap("ann", "ann")
    assert not sc._prefix_overlap("ann", "amy")


# --- people --------------------------------------------------------------

def test_match_person_finds_inflected_form():
    hit = sc._match_person(sc._words("фото с марией на море"), ["Мария", "Борис"])
    assert hit[0] == "Мария"


def test_match_person_prefers_the_name_with_more_hits():
    hit = sc._match_person(sc._words("anna smith at the lake"), ["Anna Smith", "Anna Jones"])
    assert hit == ("Anna Smith", 2)


def test_match_person_without_a_match():
    assert sc._match_person(sc._words("photo at the sea"), ["Anna", "Boris"]) is None
    assert sc._match_person(sc._words("anna"), []) is None


# --- location ------------------------------------------------------------

def test_match_location_through_a_builtin_alias():
    location = {"country": "Austria", "city": "", "province": "", "name": ""}
    assert sc._match_location(sc._words("mountains in osterreich"), location) == "Austria"


def test_match_location_prefers_the_city_label():
    location = {"country": "Austria", "city": "Vienna", "province": "", "name": ""}
    assert sc._match_location(sc._words("mountains in austria"), location) == "Vienna"


def test_match_location_reads_aliases_from_a_file(tmp_path, monkeypatch):
    aliases_file = tmp_path / "geo_aliases.json"
    aliases_file.write_text(json.dumps({"Austria": ["австрия"]}), encoding="utf-8")
    monkeypatch.setattr(sc.config, "GEO_ALIASES_FILE", aliases_file)

    location = {"country": "Austria", "city": "", "province": "", "name": ""}
    assert sc._match_location(sc._words("горы в австрии"), location) == "Austria"


def test_match_location_without_a_match():
    location = {"country": "Austria", "city": "", "province": "", "name": ""}
    assert sc._match_location(sc._words("mountains in france"), location) is None
    assert sc._location_tokens(None) == []
    assert sc._location_tokens({}) == []


# --- dates ---------------------------------------------------------------

def test_exact_day_earns_the_highest_bonus():
    bonus, label = sc._match_date(
        {"year": 2017, "months": {5}, "day": 12}, {"year": 2017, "month": 5, "day": 12}
    )
    assert bonus == sc.DATE_BONUS_DAY
    assert label == "12 May 2017"


def test_single_month_earns_more_than_a_bare_year():
    month_bonus, _ = sc._match_date(
        {"year": 2017, "months": {5}, "day": None}, {"year": 2017, "month": 5, "day": 12}
    )
    year_bonus, _ = sc._match_date(
        {"year": 2017, "months": None, "day": None}, {"year": 2017, "month": 9, "day": 3}
    )
    assert month_bonus == sc.DATE_BONUS_MONTH
    assert year_bonus == sc.DATE_BONUS_YEAR
    assert month_bonus > year_bonus


def test_season_query_shares_the_weakest_bonus():
    # Three candidate months are nearly as broad as a whole year, and the code
    # treats them that way. Pinned so a change to the branch shows up here.
    bonus, label = sc._match_date(
        {"year": None, "months": {6, 7, 8}, "day": None}, {"year": 2019, "month": 7, "day": 15}
    )
    assert bonus == sc.DATE_BONUS_YEAR
    assert label == "15 July 2019"


def test_mismatched_date_parts_do_not_match():
    assert sc._match_date({"year": 2017, "months": None, "day": None},
                          {"year": 2018, "month": 9, "day": 3}) is None
    assert sc._match_date({"year": None, "months": {5}, "day": None},
                          {"year": 2017, "month": 6, "day": 1}) is None
    assert sc._match_date({"year": None, "months": {5}, "day": 12},
                          {"year": 2017, "month": 5, "day": 13}) is None


def test_match_date_with_missing_inputs():
    assert sc._match_date(None, {"year": 2017, "month": 1, "day": 1}) is None
    assert sc._match_date({"year": 2017, "months": None, "day": None}, None) is None


# --- index loading and ranking ------------------------------------------

def _write_index(tmp_path, records):
    index_path = tmp_path / "search_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return index_path


def _record(path, description, vector, **extra):
    record = {
        "path": path,
        "description": description,
        "keywords": [],
        "people": [],
        "location": None,
        "date": None,
        "source": "photos",
        "vector": vector,
    }
    record.update(extra)
    return record


def test_search_without_an_index_reports_it(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "INDEX_PATH", tmp_path / "missing.jsonl")
    sc._INDEX_CACHE["mtime"] = None
    assert "error" in sc.search("anything")


def test_index_skips_glitched_records_and_counts_them(tmp_path, monkeypatch):
    index_path = _write_index(tmp_path, [
        _record("a.jpg", "a clean description", [1.0, 0.0]),
        _record("b.jpg", "a description with 漢字 in it", [0.0, 1.0]),
    ])
    monkeypatch.setattr(sc, "INDEX_PATH", index_path)
    sc._INDEX_CACHE["mtime"] = None

    stats = sc.stats()
    assert stats["total"] == 1
    assert stats["skippedCjk"] == 1
    assert stats["bySource"] == {"photos": 1}


def test_metadata_match_outranks_a_better_semantic_score(tmp_path, monkeypatch):
    # The query embedding is faked, so only the bonus logic decides the order:
    # "b.jpg" is semantically closer, "a.jpg" has the matching person.
    index_path = _write_index(tmp_path, [
        _record("a.jpg", "a birthday party", [1.0, 0.0], people=["Anna"]),
        _record("b.jpg", "a birthday cake", [0.0, 1.0]),
    ])
    monkeypatch.setattr(sc, "INDEX_PATH", index_path)
    monkeypatch.setattr(sc.emb, "embed", lambda texts, kind="document": np.array([[0.0, 1.0]], dtype=np.float32))
    sc._INDEX_CACHE["mtime"] = None

    result = sc.search("anna birthday", top=5)

    assert [r["path"] for r in result["results"]] == ["a.jpg", "b.jpg"]
    assert result["results"][0]["matchType"] == "person"
    assert result["results"][0]["matchLabel"] == "Anna"
    assert result["total"] == 2


def test_media_source_is_out_of_scope_by_default(tmp_path, monkeypatch):
    index_path = _write_index(tmp_path, [
        _record("a.jpg", "a photo", [1.0, 0.0]),
        _record("b.png", "an infographic", [0.0, 1.0], source="media"),
    ])
    monkeypatch.setattr(sc, "INDEX_PATH", index_path)
    monkeypatch.setattr(sc.emb, "embed", lambda texts, kind="document": np.array([[1.0, 0.0]], dtype=np.float32))
    sc._INDEX_CACHE["mtime"] = None

    default_scope = sc.search("anything", top=5)
    assert [r["path"] for r in default_scope["results"]] == ["a.jpg"]

    with_media = sc.search("anything", top=5, sources=sc.DEFAULT_SOURCES | {"media"})
    assert {r["path"] for r in with_media["results"]} == {"a.jpg", "b.png"}


def test_empty_query_returns_the_scope_size(tmp_path, monkeypatch):
    index_path = _write_index(tmp_path, [_record("a.jpg", "a photo", [1.0, 0.0])])
    monkeypatch.setattr(sc, "INDEX_PATH", index_path)
    sc._INDEX_CACHE["mtime"] = None

    result = sc.search("   ")
    assert result == {"total": 1, "results": [], "skippedCjk": 0}
