"""Vision model client: image in, free text out.

One provider (OpenRouter, OpenAI-compatible chat completions) and one stdlib-ish
dependency set (requests + Pillow). Errors are returned as {"error": ...} rather
than raised, because callers process thousands of files and a single failure must
not abort the run.

Importing this module is side-effect free: the API key is checked when a request
is actually made, so search-only tools can import it safely.
"""
import base64
import json
import mimetypes
from io import BytesIO
from pathlib import Path

import requests

import config

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Providers cap request size; large originals are downscaled before upload.
RESIZE_THRESHOLD_BYTES = 4 * 1024 * 1024
RESIZE_MAX_DIMENSION = 2048
REQUEST_TIMEOUT_SECONDS = 60

# Formats that go to the provider untouched. Everything else is converted first
# — by extension, not by the guessed mime type: mimetypes answers
# "image/x-olympus-orf" for a RAW file, which looks like an image and is not one
# as far as the provider is concerned.
UPLOADABLE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def extract_embedded_jpeg(data):
    """Return the largest embedded JPEG found in a byte blob, or None.

    RAW files (.orf and friends) usually carry a full-resolution JPEG preview.
    Pillow cannot decode RAW itself, but the preview can be recovered by
    scanning for JPEG SOI/EOI markers — no rawpy/libraw needed.
    """
    jpegs = []
    i = 0
    while True:
        start = data.find(b"\xff\xd8\xff", i)
        if start == -1:
            break
        end = data.find(b"\xff\xd9", start)
        if end == -1:
            break
        jpegs.append((start, end + 2 - start))
        i = end + 2
    if not jpegs:
        return None
    start, size = max(jpegs, key=lambda t: t[1])
    return data[start : start + size]


def _jpeg_via_pillow(data, downscale):
    """Re-encode a blob Pillow can open as JPEG, or None when it cannot open it.

    No fallback of its own: the caller decides what to do with a RAW file, and
    keeping that decision in one place is what stops the preview path from
    feeding itself.
    """
    try:
        from PIL import Image

        img = Image.open(BytesIO(data))
        if downscale:
            img.thumbnail((RESIZE_MAX_DIMENSION, RESIZE_MAX_DIMENSION))
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None


def encode_image(path):
    """Read an image and return it as a data: URL, converted where needed.

    Two separate reasons to touch the bytes, and neither is decided by size
    alone: an oversized file has to shrink, and a format the provider cannot
    read has to become one it can. A small RAW file is exactly as undecodable
    as a large one — it used to be uploaded untouched, labelled image/jpeg
    because mimetypes knows nothing about .orf.
    """
    path = Path(path)
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or ""
    oversized = len(data) > RESIZE_THRESHOLD_BYTES

    if oversized or path.suffix.lower() not in UPLOADABLE_EXTS:
        converted = _jpeg_via_pillow(data, downscale=oversized)
        if converted is None:
            # Typical for RAW: Pillow cannot decode it, but the file carries a
            # full-resolution JPEG preview that can be recovered by hand.
            preview = extract_embedded_jpeg(data)
            if preview is not None:
                converted = preview
                if oversized:
                    converted = _jpeg_via_pillow(preview, downscale=True) or preview
        if converted is not None:
            data, mime = converted, "image/jpeg"

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime or 'image/jpeg'};base64,{b64}"


def describe(image_path, prompt, reasoning=None, model=None):
    """Send one image plus prompt to the model.

    Returns {"text": str, "cost": float | None} or {"error": str}. `reasoning`
    toggles the provider's reasoning mode: None leaves the account default,
    False is markedly cheaper and faster, True reads small print in charts more
    reliably (see docs/DESIGN.md).
    """
    api_key = config.require_api_key()
    body = {
        "model": model or config.MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                ],
            }
        ],
    }
    if reasoning is not None:
        body["reasoning"] = {"enabled": bool(reasoning)}

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        return {"error": f"non-JSON response body: {resp.text[:500]}"}
    try:
        return {
            "text": payload["choices"][0]["message"]["content"],
            # "usage" is absent on some providers and explicitly null on
            # others; both used to raise straight out of the worker thread.
            "cost": (payload.get("usage") or {}).get("cost"),
        }
    except (KeyError, IndexError):
        return {"error": f"unexpected response: {json.dumps(payload)[:500]}"}
