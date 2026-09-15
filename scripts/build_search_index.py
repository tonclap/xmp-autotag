"""Build the search index: embed every description, write one JSONL line per image.

    python scripts/build_search_index.py

Each line carries the description, the keywords, and three fields the vision
model deliberately never invents — people, location and date — read straight out
of the sidecar. People and geotags are whatever a photo manager already
confirmed (Iptc4xmpExt:PersonInImage / LocationShown); the date comes from the
chain EXIF > file name > folder name. scripts/search_core.py then matches
queries against all of it, not just the embedding.

Appends and flushes after every batch, and skips paths that are already in the
index, so an interrupted run resumes instead of starting over.
"""
import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import config
import dates
import embeddings as emb
import tag_archive

INDEX_PATH = config.INDEX_PATH
BATCH_SIZE = 64

DESC_RE = re.compile(r"<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>", re.S)
SUBJ_RE = re.compile(r"<dc:subject>.*?</dc:subject>", re.S)
LI_RE = re.compile(r"<rdf:li>(.*?)</rdf:li>")
# \b[^>]*> rather than a bare ">": both elements legally carry attributes on the
# opening tag (rdf:parseType="Resource" is the common one, and the shorthand form
# puts the whole struct there and closes with "/>"). Matching only the bare tag
# silently indexed such a sidecar as having no people and no place.
PERSON_RE = re.compile(
    r"<Iptc4xmpExt:PersonInImage\b(?:[^>]*/>|[^>]*>.*?</Iptc4xmpExt:PersonInImage>)", re.S
)
LOCATION_RE = re.compile(
    r"<Iptc4xmpExt:LocationShown\b(?:[^>]*/>|[^>]*>.*?</Iptc4xmpExt:LocationShown>)", re.S
)
COUNTRY_ATTR_RE = re.compile(r'Iptc4xmpExt:CountryName="([^"]*)"')
CITY_ATTR_RE = re.compile(r'Iptc4xmpExt:City="([^"]*)"')
PROVINCE_ATTR_RE = re.compile(r'Iptc4xmpExt:ProvinceState="([^"]*)"')
LOCATION_NAME_RE = re.compile(r"<Iptc4xmpExt:LocationName>.*?<rdf:li[^>]*>(.*?)</rdf:li>", re.S)
EXIF_DATE_RE = re.compile(
    r"(?:exif:DateTimeOriginal|xmp:CreateDate|photoshop:DateCreated)="
    r'"(\d{4})-(\d{2})-(\d{2})'
)

# A description this long is not a description. Occasionally a model loops on
# self-correction and its thinking-out-loud lands in dc:description — thousands
# of characters where the normal length is a few hundred. Besides being noise in
# the results, such input breaks the ONNX embedder (its rotary cache is not
# built for that sequence length), so these records are logged and left out of
# the index. The sidecars themselves are not touched: see
# scripts/fix_script_glitches.py for repairing them deliberately.
MAX_DESCRIPTION_CHARS = 1500
SKIPPED_LOG = config.OUTPUT_DIR / "index_skipped_anomalies.jsonl"


def extract_date(content, path, root):
    """Date for one image: EXIF, then the file name, then the folder name."""
    m = EXIF_DATE_RE.search(content)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if dates.valid_year(year) and 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "source": "exif"}

    from_name = dates.from_filename(Path(path).stem)
    if from_name:
        return {**from_name, "source": "filename"}

    try:
        parts = Path(path).relative_to(root).parts[:-1]
    except ValueError:
        parts = ()
    from_folder = dates.from_folder_parts(parts)
    if from_folder:
        return {**from_folder, "source": "folder"}
    return None


def extract_people(content):
    m = PERSON_RE.search(content)
    if not m:
        return []
    return [html.unescape(name).strip() for name in LI_RE.findall(m.group(0)) if name.strip()]


def extract_location(content):
    m = LOCATION_RE.search(content)
    if not m:
        return None
    block = m.group(0)
    country = COUNTRY_ATTR_RE.search(block)
    city = CITY_ATTR_RE.search(block)
    province = PROVINCE_ATTR_RE.search(block)
    name = LOCATION_NAME_RE.search(block)
    loc = {
        "country": html.unescape(country.group(1)) if country else "",
        "city": html.unescape(city.group(1)) if city else "",
        "province": html.unescape(province.group(1)) if province else "",
        "name": html.unescape(name.group(1)).strip() if name else "",
    }
    return loc if any(loc.values()) else None


def extract(xmp_path, img_path, root):
    """Everything indexable from one sidecar."""
    content = Path(xmp_path).read_text(encoding="utf-8", errors="ignore")
    m = DESC_RE.search(content)
    if not m:
        return None, [], [], None, None
    # The text in the file is XML-escaped, and this is the point where it stops
    # being XML. Without unescaping, "Tom & Jerry" is indexed, embedded and
    # shown as "Tom &amp; Jerry" — and the web UI escapes it a second time.
    description = html.unescape(m.group(1)).strip()
    subj_block = SUBJ_RE.search(content)
    keywords = [html.unescape(kw) for kw in LI_RE.findall(subj_block.group(0))] if subj_block else []
    return (
        description,
        keywords,
        extract_people(content),
        extract_location(content),
        extract_date(content, img_path, root),
    )


def already_indexed():
    """Paths already present in the index file, tolerating truncated lines."""
    done = set()
    if not INDEX_PATH.exists():
        return done
    with INDEX_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def collect(sources):
    """Index entries for every tagged image in the given sources."""
    entries = []
    for source in sources:
        count = 0
        for img_path in tag_archive.discover(source.root, source.subdirs):
            xmp_path = img_path.with_suffix(".xmp")
            if not xmp_path.exists():
                continue
            description, keywords, people, location, date = extract(xmp_path, img_path, source.root)
            if description:
                entries.append({
                    "path": str(img_path),
                    "description": description,
                    "keywords": keywords,
                    "people": people,
                    "location": location,
                    "date": date,
                    "source": source.name,
                })
                count += 1
        print(f"  {source.name}: {count} tagged files")
    return entries


def _ensure_trailing_newline(path):
    """If a previous run died mid-line, the next append must not glue onto it."""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                with path.open("a", encoding="utf-8") as fa:
                    fa.write("\n")


def main():
    # argparse even though there are no options yet: a script that ignores
    # --help and starts a real run instead is a trap.
    argparse.ArgumentParser(
        description="Embed every tagged image and append it to the search index"
    ).parse_args()

    if not emb.model_available():
        sys.exit(
            f"embedding model not found in {emb.MODEL_DIR} - see README, "
            "'Semantic search'"
        )
    sources = tag_archive.configured_sources()
    if not sources:
        sys.exit("no archive configured - set PHOTO_ROOT (see .env.example)")

    entries = collect(sources)
    done_paths = already_indexed()
    remaining = [e for e in entries if e["path"] not in done_paths]

    anomalies = [e for e in remaining if len(e["description"]) > MAX_DESCRIPTION_CHARS]
    if anomalies:
        SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SKIPPED_LOG.open("a", encoding="utf-8") as f:
            for e in anomalies:
                f.write(json.dumps(
                    {"path": e["path"], "description_len": len(e["description"])},
                    ensure_ascii=False,
                ) + "\n")
        print(
            f"WARNING: {len(anomalies)} records longer than {MAX_DESCRIPTION_CHARS} "
            f"characters were skipped (likely damaged sidecars), see {SKIPPED_LOG}"
        )
        remaining = [e for e in remaining if len(e["description"]) <= MAX_DESCRIPTION_CHARS]

    print(f"tagged: {len(entries)}, already indexed: {len(done_paths)}, to embed: {len(remaining)}")
    if not remaining:
        print("index is up to date")
        return

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(INDEX_PATH)

    t0 = time.time()
    total_done = 0
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        for i in range(0, len(remaining), BATCH_SIZE):
            chunk = remaining[i : i + BATCH_SIZE]
            vectors = emb.embed([e["description"] for e in chunk], kind="document")
            for entry, vec in zip(chunk, vectors):
                record = dict(entry)
                record["vector"] = [round(float(x), 5) for x in vec]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # checkpoint: this batch is on disk, not just in a buffer

            total_done += len(chunk)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - total_done) / rate if rate > 0 else 0
            print(f"[{total_done}/{len(remaining)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nfinished in {time.time() - t0:.0f}s. Index: {INDEX_PATH}")


if __name__ == "__main__":
    main()
