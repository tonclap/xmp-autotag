"""Dates in free text: folder names when indexing, queries when searching.

A photo archive states its dates in three places of decreasing reliability:
EXIF inside the sidecar, a timestamp in the file name (IMG_20150407_...,
PXL_20251206_...), and prose in the folder name ("2017, 4-8 May. Trip to the
coast"). The indexer walks that chain in order; this module holds the text half
of it, plus the matching parser for queries like "photos from last summer".

Month and season words are recognised in English and Russian. Both lists are
plain data at the top of the module: adding a language means adding stems, not
touching logic. Matching is stem-based on purpose — "January", "januar",
"январь" and "января" all have to hit the same month.
"""
import re

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# (month number, regex). May is listed here for English; the Russian forms of
# "May" are short and collide with other words, so they get a separate pattern
# that is only tried when nothing else matched.
# Word boundaries are not decoration here: without them "mar" matches
# "Marbella" and "sep" matches "separate", and a folder gets dated by accident.
MONTH_PATTERNS = [
    (1, r"\bjan(?:uary)?\b|\bянвар\w*\b"),
    (2, r"\bfeb(?:ruary)?\b|\bфеврал\w*\b"),
    (3, r"\bmar(?:ch)?\b|\bмарт\w*\b"),
    (4, r"\bapr(?:il)?\b|\bапрел\w*\b"),
    (6, r"\bjun(?:e)?\b|\bиюн\w*\b"),
    (7, r"\bjul(?:y)?\b|\bиюл\w*\b"),
    (8, r"\baug(?:ust)?\b|\bавгуст\w*\b"),
    (9, r"\bsep(?:t|tember)?\b|\bсентябр\w*\b"),
    (10, r"\boct(?:ober)?\b|\bоктябр\w*\b"),
    (11, r"\bnov(?:ember)?\b|\bноябр\w*\b"),
    (12, r"\bdec(?:ember)?\b|\bдекабр\w*\b"),
]
MAY_PATTERN = r"\bmay\b|\bма[йяею]м?\b"

SEASON_PATTERNS = [
    (r"\bwinter\b|\bзим\w*\b", [12, 1, 2]),
    (r"\bspring\b|\bвесн\w*\b|\bвесенн\w*\b", [3, 4, 5]),
    (r"\bsummer\b|\bлет[оа]\b|\bлетом\b|\bлетн\w*\b", [6, 7, 8]),
    (r"\b(?:autumn|fall)\b|\bосен[ьи]\b|\bосенью\b|\bосенн\w*\b", [9, 10, 11]),
]

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
FILENAME_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?!\d)"
)
# A day standing right in front of a month name: "4-8 May" -> 4, "12 мая" -> 12.
# The leading (?<!\d) is what keeps a year out of it: without it the tail of
# "2017 May" read as day 17, and every photo in that folder was indexed — and
# labelled in the UI — as 17 May.
DAY_BEFORE_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})(?:\s*[-–]\s*\d{1,2})?\D{0,3}$")

EXACT_DATE_RE = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b")  # DD.MM.YYYY
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

MIN_YEAR, MAX_YEAR = 1900, 2100

_MONTH_RES = [(num, re.compile(pattern, re.I)) for num, pattern in MONTH_PATTERNS]
_MAY_RE = re.compile(MAY_PATTERN, re.I)
_SEASON_RES = [(re.compile(pattern, re.I), months) for pattern, months in SEASON_PATTERNS]


def valid_year(year):
    return year is not None and MIN_YEAR <= year <= MAX_YEAR


def find_month(text):
    """First month mentioned in text -> (month number, position) or (None, None).

    "First" means first in the text, not lowest month number: "December trip,
    January return" is a December folder, and the day is read from the words in
    front of the month that was actually found. May stays a fallback, tried only
    when nothing else matched, because its Russian forms are short enough to
    collide with ordinary words.
    """
    best = None
    for num, rx in _MONTH_RES:
        m = rx.search(text)
        if m and (best is None or m.start() < best[1]):
            best = (num, m.start())
    if best is not None:
        return best
    m = _MAY_RE.search(text)
    if m:
        return 5, m.start()
    return None, None


def find_year(text):
    m = YEAR_RE.search(text)
    return int(m.group(0)) if m else None


def find_season(text):
    for rx, months in _SEASON_RES:
        if rx.search(text):
            return list(months)
    return None


def from_filename(stem):
    """A date packed into a file name, e.g. IMG_20150407_123456 -> 2015-04-07."""
    m = FILENAME_DATE_RE.search(stem)
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not valid_year(year):
        return None
    return {"year": year, "month": month, "day": day}


def from_folder_parts(parts):
    """A date spelled out in folder names, e.g. "2017, 4-8 May. Trip".

    Later path segments win, so a year on the top folder can be combined with a
    month on a deeper one. A day range takes its first number ("4-8 May" -> 4),
    which is what an archive means by such a name.
    """
    year = month = day = None
    for segment in parts:
        text = segment.lower()
        found_year = find_year(text)
        if found_year is not None:
            year = found_year
        month_num, month_pos = find_month(text)
        if month_num is not None:
            month = month_num
            before = text[:month_pos]
            dm = DAY_BEFORE_MONTH_RE.search(before)
            day = int(dm.group(1)) if dm and 1 <= int(dm.group(1)) <= 31 else None
    if year is None and month is None:
        return None
    return {"year": year, "month": month, "day": day}


def parse_query(query):
    """A date constraint stated in a search query.

    Returns {"year": int|None, "months": set|None, "day": int|None} or None when
    the query says nothing about time. A season becomes a set of three months.
    """
    q = (query or "").lower()

    for rx, order in ((EXACT_DATE_RE, "dmy"), (ISO_DATE_RE, "ymd")):
        m = rx.search(q)
        if m:
            if order == "dmy":
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return {"year": year, "months": {month}, "day": day}

    year = find_year(q)
    month_num, month_pos = find_month(q)
    months = {month_num} if month_num is not None else None

    day = None
    if month_pos is not None:
        # "12 May 2017" / "12 мая 2017": a bare number right before the month.
        dm = DAY_BEFORE_MONTH_RE.search(q[:month_pos])
        if dm and 1 <= int(dm.group(1)) <= 31:
            day = int(dm.group(1))
        if day is None:
            # "May 12, 2017": the number right after the month name, as long as
            # it is not the year itself.
            after = q[month_pos:]
            dm = re.search(r"^\S+\D{0,3}(?<!\d)(\d{1,2})(?!\d)", after)
            if dm and 1 <= int(dm.group(1)) <= 31:
                day = int(dm.group(1))

    if months is None:
        season = find_season(q)
        if season:
            months = set(season)

    if year is None and months is None:
        return None
    return {"year": year, "months": months, "day": day}


def format_label(date):
    """Human label for an indexed date: "12 May 2017", "May 2017", "2017"."""
    if not date:
        return None
    year, month, day = date.get("year"), date.get("month"), date.get("day")
    name = MONTH_NAMES[month - 1] if month else None
    if name and day and year:
        return f"{day} {name} {year}"
    if name and year:
        return f"{name} {year}"
    if year:
        return str(year)
    return name
