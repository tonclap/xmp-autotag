"""Tests for scripts/dates.py: dates read out of file names, folders and queries.

Both supported languages are exercised, because the point of the module is that
one archive can hold "2017, 4-8 May. Trip" next to "ДР (январь 2024)" and a
query in either language has to reach both.
"""
import dates


# --- from_filename -------------------------------------------------------

def test_from_filename_common_camera_patterns():
    assert dates.from_filename("IMG_20150407_123456") == {"year": 2015, "month": 4, "day": 7}
    assert dates.from_filename("PXL_20251206_183012") == {"year": 2025, "month": 12, "day": 6}
    assert dates.from_filename("2019-08-30 21.04.11") == {"year": 2019, "month": 8, "day": 30}


def test_from_filename_accepts_1990s():
    assert dates.from_filename("19990607_120000") == {"year": 1999, "month": 6, "day": 7}


def test_from_filename_rejects_non_dates():
    assert dates.from_filename("DSCF0841") is None
    assert dates.from_filename("IMG_20151345_000000") is None  # month 13, day 45


# --- from_folder_parts ---------------------------------------------------

def test_folder_english_month_with_day_range_takes_first_day():
    # "4-8 May" means the trip started on the 4th; the parser says 4, not 8.
    assert dates.from_folder_parts(("2017, 4-8 may. trip to the coast",)) == {
        "year": 2017, "month": 5, "day": 4
    }


def test_folder_russian_month_stem_without_day():
    assert dates.from_folder_parts(("др (январь 2024)",)) == {
        "year": 2024, "month": 1, "day": None
    }


def test_folder_year_from_parent_month_from_child():
    assert dates.from_folder_parts(("2019", "august, seaside")) == {
        "year": 2019, "month": 8, "day": None
    }


def test_folder_without_any_date_signal():
    assert dates.from_folder_parts(("misc", "scans")) is None


def test_folder_does_not_invent_a_month_from_a_similar_word():
    # "Marbella" contains "mar", "September" logic must not fire on "separate":
    # word boundaries keep both out.
    assert dates.from_folder_parts(("marbella trip",)) is None
    assert dates.from_folder_parts(("separate scans",)) is None


# --- parse_query ---------------------------------------------------------

def test_parse_query_numeric_formats():
    assert dates.parse_query("trip 12.05.2017") == {"year": 2017, "months": {5}, "day": 12}
    assert dates.parse_query("2017-05-12 at the sea") == {"year": 2017, "months": {5}, "day": 12}


def test_parse_query_natural_language_english():
    assert dates.parse_query("photos 12 May 2017") == {"year": 2017, "months": {5}, "day": 12}
    assert dates.parse_query("May 12, 2017") == {"year": 2017, "months": {5}, "day": 12}
    assert dates.parse_query("holiday in January") == {"year": None, "months": {1}, "day": None}


def test_parse_query_natural_language_russian():
    assert dates.parse_query("фото 12 мая 2017") == {"year": 2017, "months": {5}, "day": 12}
    assert dates.parse_query("отпуск в январе") == {"year": None, "months": {1}, "day": None}


def test_parse_query_seasons_expand_to_three_months():
    assert dates.parse_query("photos from last summer")["months"] == {6, 7, 8}
    assert dates.parse_query("фото летом")["months"] == {6, 7, 8}
    assert dates.parse_query("first snow in winter")["months"] == {12, 1, 2}


def test_parse_query_year_only():
    assert dates.parse_query("photos from 2020") == {"year": 2020, "months": None, "day": None}


def test_parse_query_without_time_information():
    assert dates.parse_query("just a holiday by the sea") is None
    assert dates.parse_query("") is None


# --- format_label --------------------------------------------------------

def test_format_label_levels_of_precision():
    assert dates.format_label({"year": 2017, "month": 5, "day": 12}) == "12 May 2017"
    assert dates.format_label({"year": 2017, "month": 5, "day": None}) == "May 2017"
    assert dates.format_label({"year": 2017, "month": None, "day": None}) == "2017"
    assert dates.format_label({"year": None, "month": None, "day": None}) is None
    assert dates.format_label(None) is None


# --- a year must not be read as a day ------------------------------------

def test_year_in_front_of_the_month_is_not_a_day():
    # "2017 May" used to yield day 17 - the tail of the year - and every photo
    # in the folder was then indexed and labelled as 17 May 2017.
    assert dates.from_folder_parts(("2017 may. trip to the coast",)) == {
        "year": 2017, "month": 5, "day": None
    }
    assert dates.from_folder_parts(("2019 august",)) == {
        "year": 2019, "month": 8, "day": None
    }


def test_day_in_front_of_the_month_still_wins():
    assert dates.from_folder_parts(("2017, 8 may",))["day"] == 8
    assert dates.from_folder_parts(("8 мая 2017",))["day"] == 8


def test_query_year_before_month_has_no_day():
    assert dates.parse_query("photos from 2017 may") == {
        "year": 2017, "months": {5}, "day": None
    }


# --- find_month reports the first month in the text ----------------------

def test_find_month_returns_the_first_month_in_the_text_not_the_lowest():
    # Patterns are listed in calendar order, and the old loop returned the
    # first pattern that matched anywhere - so December..January read as
    # January, and the day was then read from the wrong side of the string.
    assert dates.find_month("december trip, january return")[0] == 12
    assert dates.find_month("сентябрь и октябрь")[0] == 9


def test_find_month_still_falls_back_to_may():
    assert dates.find_month("2017, 4-8 may")[0] == 5
    assert dates.find_month("nothing here") == (None, None)
