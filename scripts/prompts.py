"""Prompts, and the folder name that gets fed into them as a hint.

Two templates, because the two collections hold different things:

* PHOTO_PROMPT_TEMPLATE — a personal photo archive. Folder names in such an
  archive usually encode a date and an event ("2017, 4-8 May. Trip to the
  coast"), which the model cannot see in the pixels, so the folder is passed in
  as a hint — with an explicit instruction to trust the image over the hint.
* MEDIA_PROMPT_TEMPLATE — a mixed media library (infographics, memes, reference
  images, paintings). Here the folder says nothing about date or place, only
  about topic, and the description has to branch by content type: a chart wants
  its main finding stated, a meme wants the joke explained.

Both templates ask for the same output shape, which scripts/xmp.py parses:

    <one to three sentences>
    <blank line>
    <5-10 comma separated keywords>

Output language comes from TAG_LANGUAGE (see scripts/config.py) and is repeated
at the end of the prompt on purpose: with a single mention, models drift into
another language mid-answer (see docs/KNOWN_ISSUES.md).
"""
from pathlib import Path

import config

_OUTPUT_SHAPE = (
    "Then, on a separate line after a blank line, give 5-10 keywords separated "
    "by commas (singular, no hashtags). "
    "IMPORTANT: write the whole answer strictly in {language} — not a single "
    "word or character in another language or script."
)

PHOTO_PROMPT_TEMPLATE = (
    'Archive context: this photo sits in the folder "{context}", whose name '
    "usually encodes a date or an event. Treat it only as a weak hint: if what "
    "you see contradicts the folder name, trust the image.\n\n"
    "Describe this photograph in 1-3 sentences: the scene, the occasion, "
    "notable objects, the setting. Do not name people — you do not know who "
    "they are, even if a name appears in the folder name. " + _OUTPUT_SHAPE
)

MEDIA_PROMPT_TEMPLATE = (
    'This file sits in the folder "{context}" of a media library, not in a '
    "personal photo archive: do not guess a date, a place or a personal event, "
    "the folder is merely a topic hint.\n\n"
    "First decide what kind of image this is, then describe it in 1-3 "
    "sentences:\n"
    "- INFOGRAPHIC / CHART / DIAGRAM: the topic and the main finding — what "
    "data is shown, which trend, comparison or period. Restate only what is "
    "actually legible (numbers, axis labels, title).\n"
    "- PHOTOGRAPH: the scene, notable objects, the setting. Do not name people "
    "unless the subject is a widely known public figure.\n"
    "- PAINTING / ILLUSTRATION: if this is a painting, drawing, collage or any "
    "other handmade image rather than a photograph of reality — the subject, "
    "the technique or style, and the author and title if you recognise a "
    "well-known work.\n"
    "- MEME / HUMOUR: what the joke or absurdity is — what is in the image and "
    "what makes it funny, including any visible caption.\n\n" + _OUTPUT_SHAPE
)


def photo_prompt(context, language=None):
    return PHOTO_PROMPT_TEMPLATE.format(
        context=context, language=language or config.TAG_LANGUAGE
    )


def media_prompt(context, language=None):
    return MEDIA_PROMPT_TEMPLATE.format(
        context=context, language=language or config.TAG_LANGUAGE
    )


def folder_context(image_path, root):
    """Folder path of the image relative to the archive root, as a hint string.

    "<root>/2019/Summer trip/IMG_0001.jpg" -> "2019 / Summer trip". Falls back
    to the root's own name for files that sit directly in it.
    """
    path = Path(image_path)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    return " / ".join(rel.parts[:-1]) or Path(root).name
