"""Hybrid search over the index: embeddings plus exact metadata matches.

Pure semantic search cannot find a person or a place, because the vision model is
told not to guess names and not to read geography off the pixels. But a photo
manager has usually confirmed some of that by hand, and the indexer stored it —
so a query is matched three ways at once:

* cosine similarity against the description embedding (the base score);
* people, matched by word stem, so a declined or possessive form still hits;
* location, matched against city / province / country plus optional aliases;
* date, matched against the indexed date with a bonus that grows with precision.

Bonuses are large relative to cosine scores (which sit around 0.3-0.6), so an
exact metadata match always outranks a purely semantic one, while the semantic
remainder still orders results *within* the matches. Concretely: "Anna in
Austria" ranks photos matching both the person and the country first, and
"mountains in Austria" ranks Austrian photos that also look like mountains.

Shared by the CLI (search.py) and the web UI (web_search.py).
"""
import json
import re
from threading import Lock

import numpy as np

import config
import dates
import embeddings as emb

INDEX_PATH = config.OUTPUT_DIR / "search_index.jsonl"

# Searching the photo archive is the default; the media library is opt-in
# (--media on the CLI, a toggle in the web UI), because infographics and memes
# otherwise crowd out personal photos.
DEFAULT_SOURCES = frozenset({"photos"})

PERSON_MATCH_BONUS = 1.0
LOCATION_MATCH_BONUS = 0.7
# A precise date is worth more than a bare year: otherwise "photos from 2017"
# lifts all 500 photos of that year equally and drowns the semantic signal that
# still matters inside the year.
DATE_BONUS_DAY = 0.9
DATE_BONUS_MONTH = 0.6
DATE_BONUS_YEAR = 0.3

MIN_WORD_LENGTH = 3
PREFIX_MATCH_LENGTH = 4

CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Photo managers geocode in English regardless of where the photo was taken, so
# a query written in another language never matches the stored country name.
# These are examples; point GEO_ALIASES_FILE at a JSON file of your own to add
# the countries and cities your archive actually contains:
#     {"austria": ["osterreich"], "vienna": ["wien"]}
GEO_ALIASES = {
    "germany": ["deutschland"],
    "austria": ["osterreich"],
    "czech republic": ["cesko", "czechia"],
    "vienna": ["wien"],
    "munich": ["munchen"],
    "prague": ["praha"],
}


def _load_geo_aliases():
    aliases = {key: list(values) for key, values in GEO_ALIASES.items()}
    path = config.GEO_ALIASES_FILE
    if path and path.exists():
        extra = json.loads(path.read_text(encoding="utf-8"))
        for key, values in extra.items():
            aliases.setdefault(_norm(key), []).extend(
                _norm(v) for v in values
            )
    return aliases


def _norm(s):
    # "ё" folds to "е" so Russian spellings with and without it match.
    return (s or "").casefold().replace("ё", "е")


def _words(s):
    return [w for w in _WORD_RE.findall(_norm(s)) if len(w) >= MIN_WORD_LENGTH]


def _prefix_overlap(word_a, word_b, min_len=PREFIX_MATCH_LENGTH):
    """Crude stand-in for morphology: two words match if they share a prefix.

    "Anna"/"Anna's", "Мария"/"Марии" all collapse this way. It is deliberately
    not a stemmer — for browsing a photo archive by eye, a rare false positive
    on a shared four-letter prefix costs nothing.
    """
    if len(word_a) >= min_len and len(word_b) >= min_len:
        return word_a[:min_len] == word_b[:min_len]
    return word_a == word_b


def _match_person(query_words, people):
    """Best matching name from the sidecar -> (name, hit count) or None."""
    best = None
    for name in people or []:
        name_words = _words(name)
        hits = sum(1 for qw in query_words for nw in name_words if _prefix_overlap(qw, nw))
        if hits and (best is None or hits > best[1]):
            best = (name, hits)
    return best


def _location_tokens(location, aliases=None):
    if not location:
        return []
    aliases = aliases if aliases is not None else _load_geo_aliases()
    tokens = []
    for field in ("city", "province", "country"):
        value = location.get(field)
        if value:
            tokens.append(_norm(value))
            tokens.extend(aliases.get(_norm(value), []))
    tokens.extend(_words(location.get("name", "")))
    return tokens


def _match_location(query_words, location, aliases=None):
    """Label of the matched place, preferring the most specific field."""
    tokens = _location_tokens(location, aliases)
    if not tokens:
        return None
    for qw in query_words:
        for token in tokens:
            token_words = _words(token) or [token]
            if any(_prefix_overlap(qw, tw) for tw in token_words):
                return location.get("city") or location.get("province") or location.get("country")
    return None


def _parse_query_date(query):
    return dates.parse_query(query)


def _format_date_label(date):
    return dates.format_label(date)


def _match_date(query_date, record_date):
    """(bonus, label) when the record satisfies the query's date, else None."""
    if not query_date or not record_date:
        return None
    q_year, q_months, q_day = query_date.get("year"), query_date.get("months"), query_date.get("day")
    r_year, r_month, r_day = record_date.get("year"), record_date.get("month"), record_date.get("day")

    if q_year is not None and r_year != q_year:
        return None
    if q_months is not None and r_month not in q_months:
        return None
    if q_day is not None and r_day != q_day:
        return None

    if q_day is not None:
        bonus = DATE_BONUS_DAY
    elif q_months is not None and len(q_months) == 1:
        bonus = DATE_BONUS_MONTH
    else:
        # A bare year, and also a season: three candidate months are nearly as
        # broad as a year, so they share the weakest bonus.
        bonus = DATE_BONUS_YEAR
    return bonus, _format_date_label(record_date)


# Brute-force cosine over the whole index: ten thousand vectors take
# milliseconds, so there is no reason for an approximate index. Cached by file
# mtime to avoid re-reading the JSONL on every keystroke.
_INDEX_CACHE = {"mtime": None, "vectors": None, "meta": None, "skipped_cjk": 0}
# The web UI is a ThreadingHTTPServer, so several requests read this cache at
# once. Without the lock a request could pick up the vectors of one generation
# and the metadata of the next, and rows would not line up with paths.
_INDEX_LOCK = Lock()


def _load_index():
    meta, vectors, skipped_cjk = [], [], 0
    with INDEX_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record["description"] + " " + " ".join(record.get("keywords") or [])
            if CJK_RE.search(text):
                # Known model glitch (docs/KNOWN_ISSUES.md): stray CJK characters
                # in text that should not contain any. Counted and reported
                # rather than silently dropped.
                skipped_cjk += 1
                continue
            meta.append({
                "path": record["path"],
                "description": record["description"],
                "keywords": record.get("keywords") or [],
                "people": record.get("people") or [],
                "location": record.get("location"),
                "date": record.get("date"),
                "source": record.get("source") or "photos",
            })
            vectors.append(record["vector"])
    if not vectors:
        return None, [], skipped_cjk
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return matrix / norms, meta, skipped_cjk


def get_index():
    """(unit vectors, metadata, skipped count); vectors is None when empty."""
    if not INDEX_PATH.exists():
        return None, [], 0
    mtime = INDEX_PATH.stat().st_mtime
    with _INDEX_LOCK:
        if _INDEX_CACHE["mtime"] != mtime:
            vectors, meta, skipped = _load_index()
            _INDEX_CACHE.update(mtime=mtime, vectors=vectors, meta=meta, skipped_cjk=skipped)
        return _INDEX_CACHE["vectors"], _INDEX_CACHE["meta"], _INDEX_CACHE["skipped_cjk"]


def stats():
    vectors, meta, skipped = get_index()
    by_source = {}
    for m in meta:
        by_source[m["source"]] = by_source.get(m["source"], 0) + 1
    return {
        "indexed": vectors is not None,
        "total": len(meta),
        "skippedCjk": skipped,
        "bySource": by_source,
    }


def search(query, top=48, sources=DEFAULT_SOURCES):
    """Ranked results for one query.

    Returns {"total", "results", "skippedCjk"}, or {"error": ...} when there is
    no index yet. Each result carries the score and, when a metadata match drove
    it, matchType ("person" / "location" / "date") and matchLabel.
    """
    vectors, meta, skipped = get_index()
    if vectors is None:
        return {"error": "no index yet - run: python scripts/build_search_index.py"}

    query = (query or "").strip()
    in_scope = [i for i, m in enumerate(meta) if m["source"] in sources]
    if not query:
        return {"total": len(in_scope), "results": [], "skippedCjk": skipped}

    q = emb.embed([query], kind="query")[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    semantic_scores = vectors @ q

    query_words = _words(query)
    query_date = _parse_query_date(query)
    aliases = _load_geo_aliases()

    scored = []
    for i in in_scope:
        m = meta[i]
        score = float(semantic_scores[i])
        match_type = match_label = None

        person_hit = _match_person(query_words, m["people"]) if query_words else None
        if person_hit:
            # Tiny per-hit increment so "first last" outranks a single-name hit.
            score += PERSON_MATCH_BONUS + 0.01 * person_hit[1]
            match_type, match_label = "person", person_hit[0]

        location_hit = _match_location(query_words, m["location"], aliases) if query_words else None
        if location_hit:
            score += LOCATION_MATCH_BONUS
            if match_type is None:
                match_type, match_label = "location", location_hit

        date_hit = _match_date(query_date, m["date"])
        if date_hit:
            bonus, label = date_hit
            score += bonus
            if match_type is None:
                match_type, match_label = "date", label

        scored.append((i, score, match_type, match_label))

    scored.sort(key=lambda t: -t[1])
    results = []
    for i, score, match_type, match_label in scored[: max(0, top)]:
        m = meta[i]
        results.append({
            "path": m["path"],
            "description": m["description"],
            "keywords": m["keywords"],
            "score": score,
            "matchType": match_type,
            "matchLabel": match_label,
            "source": m["source"],
        })
    return {"total": len(in_scope), "results": results, "skippedCjk": skipped}
