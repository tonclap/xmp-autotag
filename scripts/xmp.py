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
from pathlib import Path
from xml.sax.saxutils import escape

import config

DC_NS = "http://purl.org/dc/elements/1.1/"
TOOL_NAME = "xmp-autotag"

# Overridden per run by tag_archive.py / retry_failed.py so each run keeps its
# own backups.
BACKUP_DIR = config.OUTPUT_DIR / "xmp_backup"


def parse_text(text):
    """Split raw model output into (description, keywords).

    The documented shape is "sentences\\n\\nkeyword, keyword, ...", but models
    sometimes omit the blank line; in that case the last line is taken as the
    keyword list if it looks like one. If no keyword list can be found, the
    whole text becomes the description and keywords come back empty — callers
    treat that as a parse failure and skip the file.
    """
    description, _, keywords_line = (text or "").rpartition("\n\n")
    if not description:
        description, keywords_line = keywords_line, ""
    keywords = [kw.strip().rstrip(".") for kw in keywords_line.split(",") if kw.strip()]
    if not keywords:
        description, _, last_line = (text or "").rpartition("\n")
        if "," in last_line:
            keywords = [kw.strip().rstrip(".") for kw in last_line.split(",") if kw.strip()]
        else:
            description = text or ""
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


def merge_existing(xmp_path, subject_xml, desc_xml):
    """Insert the fields into an existing sidecar.

    Returns (backup_path, None) on success, or (None, reason) when the file was
    left untouched. The original is copied into BACKUP_DIR before any write.
    """
    content = xmp_path.read_text(encoding="utf-8")

    if "<dc:subject>" in content:
        return None, "already has dc:subject - skipped, not overwriting"

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"{xmp_path.stem}__{abs(hash(str(xmp_path)))}.xmp"
    shutil.copy2(xmp_path, backup_path)

    # Drop the old dc:description first, otherwise the file ends up with two of
    # them. Safe by rule 2 in the module docstring: anything still here is
    # camera or importer noise, since a hand-tagged file would have dc:subject
    # and would have been skipped above.
    content = re.sub(r"\s*<dc:description>.*?</dc:description>", "", content, count=1, flags=re.S)

    if "xmlns:dc=" not in content:
        marker = '<rdf:Description rdf:about=""'
        if marker not in content:
            return None, "rdf:Description opening tag not found - skipped"
        content = content.replace(marker, f'{marker}\n    xmlns:dc="{DC_NS}"', 1)

    # rdf:Description nests: face regions and LocationShown carry their own
    # inner rdf:Description. The LAST </rdf:Description> in the document always
    # closes the outer one, because in valid XML children close before parents.
    closing = "</rdf:Description>"
    idx = content.rfind(closing)
    if idx != -1:
        content = (
            content[:idx]
            + f"{subject_xml}\n{desc_xml}\n{closing}"
            + content[idx + len(closing) :]
        )
    else:
        # No closing tag at all means the top-level element is self-closing
        # (<rdf:Description ... />) and has no children. Turn "/>" into an open
        # tag, the new fields, and an explicit close.
        marker = '<rdf:Description rdf:about=""'
        start = content.find(marker)
        if start == -1:
            return None, "rdf:Description opening tag not found - skipped"
        selfclose_idx = content.find("/>", start)
        if selfclose_idx == -1:
            return None, "self-closing rdf:Description not found - skipped"
        content = (
            content[:selfclose_idx]
            + f">\n   {subject_xml}\n   {desc_xml}\n  </rdf:Description>"
            + content[selfclose_idx + 2 :]
        )

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
