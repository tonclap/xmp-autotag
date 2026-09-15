"""Writing model output into XMP sidecars.

Two rules shape this module, and both come from running it over a real archive
rather than from the spec:

1. An existing sidecar is edited as *text*, not parsed and re-serialised. Photo
   managers store things there that matter more than our keywords — face
   regions (mwg-rs:Regions), confirmed people (Iptc4xmpExt:PersonInImage),
   geotags (Iptc4xmpExt:LocationShown), ratings — and any XML round-trip
   reorders or drops attributes. A targeted string insert cannot.
2. "Already tagged" is decided by dc:subject only. dc:description in the wild is
   almost always camera or importer noise ("OLYMPUS DIGITAL CAMERA",
   "Processed with ..."), so it is safe to replace; dc:subject is the field a
   human would have filled in by hand, so its presence means hands off.

Nothing here touches the image file itself — only the .xmp next to it.
"""
import re
import shutil
from hashlib import sha1
from pathlib import Path
from xml.sax.saxutils import escape

import config

DC_NS = "http://purl.org/dc/elements/1.1/"
TOOL_NAME = "xmp-autotag"

# Overridden per run by tag_archive.py / retry_failed.py so each run keeps its
# own backups.
BACKUP_DIR = config.OUTPUT_DIR / "xmp_backup"


# A keyword is a word or a short phrase. A model that answers with a second
# paragraph of prose instead of a keyword list produces comma-separated *clauses*
# — and without this limit they were written into dc:subject as if they were
# keywords, with the sentence they came from cut out of the description. Since
# dc:subject is what marks a file as done, that damage is permanent: a re-run
# skips the file. Rejecting the whole answer is the cheaper mistake — the file is
# logged as a parse failure and retry_failed.py asks again.
MAX_KEYWORD_WORDS = 3


def _as_keywords(line):
    """Comma-separated items of `line`, or [] when it is prose, not a list."""
    items = [kw.strip().rstrip(".") for kw in line.split(",") if kw.strip()]
    if len(items) < 2:
        return []
    if any(len(kw.split()) > MAX_KEYWORD_WORDS for kw in items):
        return []
    return items


def parse_text(text):
    """Split raw model output into (description, keywords).

    The documented shape is "sentences\\n\\nkeyword, keyword, ...", but models
    sometimes omit the blank line; in that case the last line is taken as the
    keyword list if it looks like one. If no keyword list can be found, the
    whole text becomes the description and keywords come back empty — callers
    treat that as a parse failure and skip the file.
    """
    text = text or ""
    description, _, keywords_line = text.rpartition("\n\n")
    keywords = _as_keywords(keywords_line) if description else []
    if not keywords:
        head, _, last_line = text.rpartition("\n")
        keywords = _as_keywords(last_line) if head else []
        description = head if keywords else text
    return description.strip(), keywords


def build_fields(description, keywords):
    """Return (dc:subject XML, dc:description XML) with text properly escaped."""
    subj_items = "".join(f"<rdf:li>{escape(kw)}</rdf:li>" for kw in keywords)
    subject = f"<dc:subject><rdf:Bag>{subj_items}</rdf:Bag></dc:subject>"
    desc = (
        "<dc:description><rdf:Alt>"
        f'<rdf:li xml:lang="x-default">{escape(description)}</rdf:li>'
        "</rdf:Alt></dc:description>"
    )
    return subject, desc


_RDF_TAG_RE = re.compile(r"<rdf:Description\b[^>]*>|</rdf:Description>", re.S)


def top_level_blocks(content):
    """Every TOP-LEVEL rdf:Description in the document, in order.

    Each entry is a dict with the offsets of its opening tag and of its closing
    tag (None when the element is self-closing). Nesting is tracked with a depth
    counter: face regions and LocationShown carry inner rdf:Description
    elements, and those must never be picked as an insertion point.

    Why a scan and not "the last closing tag": rdf:RDF is allowed to hold
    several sibling rdf:Description blocks — one per schema is exactly how Adobe
    products write a sidecar — and each block declares its own namespaces. The
    last closing tag then belongs to the last sibling, which normally does not
    declare xmlns:dc; writing dc:* fields there produces a sidecar no XML parser
    will read.
    """
    blocks = []
    depth = 0
    opening = None
    for m in _RDF_TAG_RE.finditer(content):
        tag = m.group(0)
        if tag.startswith("</"):
            depth = max(0, depth - 1)
            if depth == 0 and opening is not None:
                blocks.append({
                    "open_start": opening.start(),
                    "open_end": opening.end(),
                    "close_start": m.start(),
                    "self_closing": False,
                })
                opening = None
        elif tag.endswith("/>"):
            if depth == 0:
                blocks.append({
                    "open_start": m.start(),
                    "open_end": m.end(),
                    "close_start": None,
                    "self_closing": True,
                })
        else:
            if depth == 0:
                opening = m
            depth += 1
    return blocks


def _pick_target_block(content, blocks):
    """The block the dc:* fields must go into, and whether it needs xmlns:dc.

    Preference order: the block that already declares xmlns:dc, then any block
    if the declaration sits on an ancestor (rdf:RDF or x:xmpmeta), then the
    first block — which is the one that gets the declaration added.
    """
    for block in blocks:
        if "xmlns:dc=" in content[block["open_start"] : block["open_end"]]:
            return block, False
    declared_on_ancestor = "xmlns:dc=" in content[: blocks[0]["open_start"]]
    return blocks[0], not declared_on_ancestor


def merge_existing(xmp_path, subject_xml, desc_xml):
    """Insert the fields into an existing sidecar.

    Returns (backup_path, None) on success, or (None, reason) when the file was
    left untouched. The original is copied into BACKUP_DIR before any write.
    """
    content = xmp_path.read_text(encoding="utf-8")

    if "<dc:subject>" in content:
        return None, "already has dc:subject - skipped, not overwriting"

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"{xmp_path.stem}__{_path_digest(xmp_path)}.xmp"
    shutil.copy2(xmp_path, backup_path)

    # Drop the old dc:description first, otherwise the file ends up with two of
    # them. Safe by rule 2 in the module docstring: anything still here is
    # camera or importer noise, since a hand-tagged file would have dc:subject
    # and would have been skipped above.
    content = re.sub(r"\s*<dc:description>.*?</dc:description>", "", content, count=1, flags=re.S)

    blocks = top_level_blocks(content)
    if not blocks:
        if re.search(r"<rdf:Description\b", content):
            # An opening tag with neither a closing tag nor a self-closing form:
            # the sidecar is truncated or malformed. Say so, do not guess.
            return None, "self-closing rdf:Description not found - skipped"
        return None, "rdf:Description opening tag not found - skipped"

    block, needs_ns = _pick_target_block(content, blocks)

    if block["self_closing"]:
        # No children, so there is no closing tag to insert before: turn the
        # trailing "/>" into an open tag, the new fields, and an explicit close.
        open_tag = content[block["open_start"] : block["open_end"]]
        if needs_ns:
            open_tag = _with_dc_namespace(open_tag)
        replacement = (
            f"{open_tag[:-2]}>\n   {subject_xml}\n   {desc_xml}\n  </rdf:Description>"
        )
        return _write(
            xmp_path,
            content[: block["open_start"]] + replacement + content[block["open_end"] :],
            backup_path,
        )

    tail = content[block["close_start"] :]
    head = content[: block["close_start"]]
    if needs_ns:
        open_tag = content[block["open_start"] : block["open_end"]]
        head = (
            content[: block["open_start"]]
            + _with_dc_namespace(open_tag)
            + content[block["open_end"] : block["close_start"]]
        )
    return _write(xmp_path, f"{head}{subject_xml}\n{desc_xml}\n{tail}", backup_path)


def _with_dc_namespace(open_tag):
    """Add xmlns:dc to an rdf:Description opening tag (self-closing or not)."""
    cut = -2 if open_tag.endswith("/>") else -1
    return f'{open_tag[:cut]}\n    xmlns:dc="{DC_NS}"{open_tag[cut:]}'


def _path_digest(path):
    """Short, stable digest of a path, used to name its backup.

    sha1 rather than hash(): the built-in is salted per process, so the same
    sidecar got a different backup name on every run and a backup could not be
    traced back to the file it came from.
    """
    return sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _write(xmp_path, content, backup_path):
    xmp_path.write_text(content, encoding="utf-8")
    return backup_path, None


def create_new(xmp_path, subject_xml, desc_xml, model=None):
    """Write a minimal sidecar for an image that had none.

    Filename convention: same basename as the image, .xmp extension — what photo
    managers look for.
    """
    toolkit = f"{TOOL_NAME}/{model or config.MODEL}"
    content = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{escape(toolkit)}">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="{DC_NS}">
   {subject_xml}
   {desc_xml}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    xmp_path.write_text(content, encoding="utf-8")


def replace_fields(xmp_path, subject_xml, desc_xml):
    """Overwrite dc:subject and dc:description that are already present.

    Used when re-tagging a file whose stored text turned out to be broken (see
    scripts/fix_script_glitches.py). Everything else in the sidecar — including
    fields a photo manager added after the first tagging run — is left alone.
    """
    content = xmp_path.read_text(encoding="utf-8")
    if "<dc:subject>" not in content or "<dc:description>" not in content:
        return False, "no existing dc:subject/dc:description to replace"
    # Function replacements, not string ones: re treats backslash escapes in a
    # replacement string, and a description can legitimately contain one.
    new_content, n1 = _SUBJECT_RE.subn(lambda _m: subject_xml, content, count=1)
    new_content, n2 = _DESC_RE.subn(lambda _m: desc_xml, new_content, count=1)
    if n1 != 1 or n2 != 1:
        return False, f"unexpected replacement count subj={n1} desc={n2}"
    xmp_path.write_text(new_content, encoding="utf-8")
    return True, None


_SUBJECT_RE = re.compile(r"<dc:subject>.*?</dc:subject>", re.S)
_DESC_RE = re.compile(r"<dc:description>.*?</dc:description>", re.S)


def has_subject(xmp_path):
    """True if the sidecar exists and already carries dc:subject."""
    path = Path(xmp_path)
    if not path.exists():
        return False
    try:
        return "<dc:subject>" in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
