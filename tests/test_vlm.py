"""Tests for scripts/vlm.py: response handling and the RAW preview scan.

No network: requests.post is replaced with a stub returning a canned payload,
and encode_image with a constant, so nothing is read from disk either. What is
pinned here is the shape of the answers the provider actually sends back — a
missing usage block, a null content, an error body — because each of those runs
in a worker thread where an unexpected exception costs a paid call.
"""
import pytest

import vlm


class _Response:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def stub_provider(monkeypatch):
    """Answer every request with a payload of the test's choosing."""
    monkeypatch.setattr(vlm.config, "API_KEY", "sk-or-test")
    monkeypatch.setattr(vlm, "encode_image", lambda path: "data:image/jpeg;base64,AA")

    def install(response):
        monkeypatch.setattr(vlm.requests, "post", lambda *a, **kw: response)

    return install


def test_describe_returns_text_and_cost(stub_provider):
    stub_provider(_Response({
        "choices": [{"message": {"content": "A dog.\n\ndog"}}],
        "usage": {"cost": 0.00004},
    }))
    assert vlm.describe("a.jpg", "prompt") == {"text": "A dog.\n\ndog", "cost": 0.00004}


@pytest.mark.parametrize("payload", [
    {"choices": [{"message": {"content": "A dog."}}]},                # no usage key
    {"choices": [{"message": {"content": "A dog."}}], "usage": None},  # usage: null
])
def test_describe_survives_a_missing_usage_block(stub_provider, payload):
    # Both shapes come back from real providers, and both used to raise
    # AttributeError out of the worker thread - the call was paid for and the
    # image stayed untagged.
    stub_provider(_Response(payload))
    assert vlm.describe("a.jpg", "prompt") == {"text": "A dog.", "cost": None}


def test_describe_reports_a_null_content_as_text_none(stub_provider):
    # A content filter refusal arrives as 200 OK with an empty body; the caller
    # distinguishes it from an error by text being None.
    stub_provider(_Response({"choices": [{"message": {"content": None}}]}))
    assert vlm.describe("a.jpg", "prompt")["text"] is None


def test_describe_reports_an_http_error(stub_provider):
    stub_provider(_Response({}, status_code=429, text="rate limited"))
    assert vlm.describe("a.jpg", "prompt")["error"].startswith("HTTP 429")


def test_describe_reports_an_unexpected_payload(stub_provider):
    stub_provider(_Response({"choices": []}))
    assert "unexpected response" in vlm.describe("a.jpg", "prompt")["error"]


# --- extract_embedded_jpeg ----------------------------------------------

def _jpeg(payload):
    return b"\xff\xd8\xff" + payload + b"\xff\xd9"


def test_extract_embedded_jpeg_takes_the_largest_preview():
    # A RAW file carries a thumbnail and a full-size preview; the big one is
    # the only one worth sending to the model.
    blob = b"RAWHEADER" + _jpeg(b"small") + b"\x00\x00" + _jpeg(b"a much larger preview")
    assert vlm.extract_embedded_jpeg(blob) == _jpeg(b"a much larger preview")


def test_extract_embedded_jpeg_returns_none_without_a_preview():
    assert vlm.extract_embedded_jpeg(b"no jpeg markers here") is None


# --- encode_image: conversion is decided by format, not only by size ----

def _jpeg_bytes(size=(60, 40)):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, "red").save(buf, format="JPEG")
    return buf.getvalue()


def _decoded(data_url):
    import base64

    header, _, payload = data_url.partition(",")
    return header, base64.b64decode(payload)


def test_a_small_raw_file_is_converted_to_its_preview(tmp_path):
    # mimetypes answers "image/x-olympus-orf" for .orf, which passes for an
    # image type and is not one to the provider. Under the size threshold the
    # file used to be uploaded as raw bytes labelled image/jpeg.
    raw = tmp_path / "a.orf"
    preview = _jpeg_bytes()
    raw.write_bytes(b"ORFHEADER" + preview + b"trailing bytes")

    header, payload = _decoded(vlm.encode_image(raw))

    assert header == "data:image/jpeg;base64"
    assert payload == preview


def test_an_ordinary_small_jpeg_is_sent_untouched(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(_jpeg_bytes())

    header, payload = _decoded(vlm.encode_image(photo))

    assert header == "data:image/jpeg;base64"
    assert payload == photo.read_bytes()  # no re-encoding, no quality loss


def test_a_png_keeps_its_own_type(tmp_path):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (10, 10), "blue").save(buf, format="PNG")
    photo = tmp_path / "a.png"
    photo.write_bytes(buf.getvalue())

    header, payload = _decoded(vlm.encode_image(photo))

    assert header == "data:image/png;base64"
    assert payload == photo.read_bytes()


def test_an_oversized_image_is_downscaled(tmp_path, monkeypatch):
    from io import BytesIO

    from PIL import Image

    monkeypatch.setattr(vlm, "RESIZE_THRESHOLD_BYTES", 1024)
    monkeypatch.setattr(vlm, "RESIZE_MAX_DIMENSION", 64)
    buf = BytesIO()
    Image.new("RGB", (900, 600), "red").save(buf, format="PNG")
    photo = tmp_path / "big.png"
    photo.write_bytes(buf.getvalue())

    header, payload = _decoded(vlm.encode_image(photo))

    assert header == "data:image/jpeg;base64"
    assert max(Image.open(BytesIO(payload)).size) <= 64


def test_a_raw_file_without_a_preview_is_not_mislabelled(tmp_path):
    # Nothing can be done for this one, and it is honest about that: the
    # provider gets the real type rather than a JPEG label on RAW bytes.
    raw = tmp_path / "a.orf"
    raw.write_bytes(b"no jpeg markers in here at all")

    header, payload = _decoded(vlm.encode_image(raw))

    assert header != "data:image/jpeg;base64"
    assert payload == raw.read_bytes()
