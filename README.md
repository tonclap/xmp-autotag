# xmp-autotag

Describe every photo in an archive with a vision model, and write the result into
**standard XMP sidecars** — `dc:subject` (keywords) and `dc:description` (one to
three sentences about the scene) — right next to the images. Then search the
archive in plain language, and find near-duplicate frames.

The point is where the tags go. They are not kept in a database owned by this
tool: they land in the open fields every photo manager and metadata reader
already understands, so the archive stays searchable even if you throw this
repository away tomorrow.

```
python scripts/probe.py                    # look at 10 answers, write nothing
python scripts/tag_archive.py --limit 20   # tag a first batch for real
python scripts/tag_archive.py              # tag the rest (resumable)
python scripts/build_search_index.py       # embed the descriptions
python scripts/web_search.py               # http://localhost:8766
```

## What it does

* **Tags** — walks the archive, sends each image to a vision model with the folder
  name as a weak context hint, and merges the answer into the sidecar. Existing
  metadata is preserved: face regions, confirmed people, geotags and ratings are
  never touched, and an image that already has `dc:subject` is left alone.
* **Searches** — embeds the descriptions locally and ranks a plain-language query
  against them, plus exact matches on the people, places and dates already in the
  sidecars. "Anna in Austria", "first snow", "12 May 2017" and "photos from last
  summer" all work, in the web UI or on the command line.
* **Finds near-duplicates** — clusters visually similar frames within a folder, the
  burst shots that byte-level duplicate finders never catch. It reports them; it
  never deletes anything.

## What it does not do

* It is not a photo manager: no viewing, syncing, face recognition, ratings, backup.
* It does not reorganise your folders or rename anything.
* It does not write into the image files themselves — only into `.xmp` sidecars.
* It is not a backup tool. Sidecars are modified in place (with a backup copy of
  each one before the first edit), so run it on an archive you have a backup of.

## Requirements

* Python 3.10 or newer, `pip install -r requirements.txt`
* An [OpenRouter](https://openrouter.ai) API key — for tagging only
* An ONNX text embedding model on disk — for search only (see below)

Tagging and searching are independent. Browsing an already tagged archive needs no
API key and no network; building the index needs no API key either.

## Setup

```bash
git clone https://github.com/<you>/xmp-autotag.git
cd xmp-autotag
pip install -r requirements.txt
cp .env.example .env     # then edit it
```

At minimum, set `PHOTO_ROOT` to your archive and `OPENROUTER_API_KEY` to your key.
`TAG_LANGUAGE` decides which language the descriptions are written in (default
English) — it goes into the prompt as plain text, so `German` or `Brazilian
Portuguese` work as well as `English`.

Every setting can also come from a real environment variable, which wins over
`.env`:

```bash
PHOTO_ROOT=/mnt/photos python scripts/tag_archive.py --limit 5
```

## Tagging

```bash
python scripts/probe.py --limit 10          # dry run: prints answers, writes nothing
python scripts/probe.py --reasoning         # compare with reasoning mode on
python scripts/tag_archive.py --limit 20    # a first real batch
python scripts/tag_archive.py               # the whole archive
python scripts/retry_failed.py              # second pass over the failures
```

Start with `probe.py`. It calls the model on a sample spread across the archive
and prints what comes back, so you can judge the language, the level of detail and
whether reasoning mode is worth its price on *your* material before spending
anything at scale.

`tag_archive.py` is idempotent and interruptible: an image counts as done once its
sidecar has `dc:subject`, so Ctrl+C and restart is a normal way to work. Every
file is recorded in `output/tag_photos_log.jsonl`, and `retry_failed.py` reads
that log to retry rate limits, oversized uploads and malformed answers.

An optional second collection with different content — infographics, memes,
reference images — can be tagged with a prompt suited to it:

```bash
python scripts/tag_archive.py --source media
```

Set `MEDIA_ROOT` and `MEDIA_SUBDIRS` for that; only the listed subfolders are
walked, because such a library usually also holds things there is no point in
describing.

### Cost and speed

On the archive this was built against — 10,270 images, a decade and a half of a
family archive — a full tagging run cost **$0.90** and took a few hours with eight
workers, using `qwen/qwen3.7-flash` with reasoning off. That is roughly
$0.00005 per image. With reasoning on, the same archive would have cost about
$2.50-3.00, with no visible gain in description quality for photographs — which is
why reasoning is off by default for photos and on for the media library, where it
does measurably help with reading numbers off charts. See
[docs/DESIGN.md](docs/DESIGN.md).

## Semantic search

Indexing and searching need a sentence embedding model exported to ONNX, in
`EMBEDDING_MODEL_DIR`:

```
<EMBEDDING_MODEL_DIR>/
  model_quantized.onnx    # must expose a "sentence_embedding" output
  tokenizer.json
```

Nothing is downloaded automatically. Any retrieval model with a pooled sentence
output works; the prefixes in `scripts/embeddings.py` (`task: search result |
query:` / `title: none | text:`) follow the EmbeddingGemma convention, so a model
from a different family may want different ones. Quantised int8 models are
recommended: an archive of ten thousand descriptions embeds on a CPU in minutes
and searches in milliseconds.

```bash
python scripts/build_search_index.py        # resumable, appends to output/search_index.jsonl
python scripts/search.py "kids in the snow" --top 10
python scripts/web_search.py                # thumbnails, badges, duplicates tab
python scripts/find_near_duplicates.py --threshold 0.93
```

The web UI binds to `127.0.0.1` only and has no authentication: it serves images
straight off your disk. Do not put it on a public interface.

Queries are matched against dates and place names in English and Russian out of
the box. Photo managers geocode in English regardless of where a photo was taken,
so if you search in another language, add your own aliases:

```bash
# geo_aliases.json
{"austria": ["österreich"], "vienna": ["wien"]}
```

and point `GEO_ALIASES_FILE` at the file.

## Tests

```bash
python -m pytest tests -q
```

121 tests, all offline: no network, no model, no archive. They run on a fresh clone
with no `.env` at all, and cover the parts where a mistake is expensive — editing
sidecars without damaging existing metadata, the date chain, the ranking bonuses,
the duplicate clustering.

## Documentation

* [docs/DESIGN.md](docs/DESIGN.md) — why sidecars and not a database, why the merge
  is textual, why reasoning is off, what the census of a real archive showed
* [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — the limits that were accepted
  rather than fixed, and why

## Licence

MIT — see [LICENSE](LICENSE).
