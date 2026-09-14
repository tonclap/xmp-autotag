# Design notes

Why this tool is shaped the way it is. Most of these decisions were forced by a
real archive — roughly 10,000 images spanning 1998-2026, already managed by a
commercial photo manager — rather than chosen on paper.

## Tags go into open XMP fields, not into a database of our own

A tagger that keeps its output in its own store creates a dependency: the archive
is only searchable through the tool that wrote the tags. Writing into
`dc:subject` and `dc:description` in the sidecar means any photo manager, any
metadata reader and any future tool sees the work, and this repository can be
deleted without losing it.

The cost of that choice is that we are writing into files that something else
owns. Hence the safety rules below.

## A census came first, and it changed the plan

Before writing a single field, all 2,260 sidecars of that archive were counted:

| What | Where it was found |
| --- | --- |
| Confirmed people (`Iptc4xmpExt:PersonInImage`, `mwg-rs` face regions) | 1,636 sidecars (72%) |
| Geotags (`Iptc4xmpExt:LocationShown`) | 794 (35%) |
| `dc:subject` — the standard keyword field | **2 files**, and neither was hand-written by the owner |
| `dc:description` | 318 (14%), of which 316 were camera noise: `OLYMPUS DIGITAL CAMERA`, `SAMSUNG DIGITAL CAMERA` and similar |

Three conclusions, all of which are encoded in the code:

1. **The keyword fields are free.** There was no hand-made keyword work to
   destroy, which is what makes bulk writing acceptable at all.
2. **The valuable metadata is people and places** — and it is exactly what a
   vision model must not overwrite or invent. The prompt forbids naming people,
   and the merge never touches those fields.
3. **Metadata lived in sidecars only**, not embedded in the JPEGs: images without
   a sidecar had no XMP packet inside the binary either. So a missing sidecar is
   really missing data, and the tool creates a new one rather than writing into
   the image.

Your archive may differ. Check before a bulk run: if `dc:subject` is full of
work you did by hand, this tool's assumption does not hold for you (it will skip
those files, but the assumption behind replacing `dc:description` is weaker too).

## The merge is textual, not an XML round-trip

An existing sidecar is edited by inserting strings at a known position. Parsing
and re-serialising with any XML library reorders attributes, rewrites namespace
prefixes and drops formatting — and the things most likely to be mangled are the
face regions and geotags that matter more than our keywords.

Two shapes that a naive insert gets wrong, both found in the wild and both
covered by tests:

* `<rdf:Description rdf:about="" .../>` — self-closing, no children. There is no
  closing tag to insert before, so the `/>` has to be expanded first.
* A **nested** `rdf:Description` inside `mwg-rs:Regions` or `LocationShown`. The
  *last* `</rdf:Description>` in the document closes the outer element, because
  in valid XML children close before parents — so that is the anchor, not the
  first one.

## "Already tagged" means `dc:subject`, and only that

`dc:description` is treated as replaceable (per the census above: it is almost
always camera noise), while `dc:subject` is treated as sacred. One field decides
idempotency, which is what makes an interrupted run safe to restart: the state
lives in the archive itself, not in a log that can drift away from it.

Before the first edit of any existing sidecar, a copy goes into
`output/xmp_backup_<source>/`. New sidecars are not backed up — there was nothing
there.

## Reasoning mode: off for photographs, on for charts

The provider's reasoning mode costs roughly five times more per image and is
slower. On photographs it produced no visible improvement in the description, so
it is off for the photo archive: $0.90 versus around $2.50-3.00 for the same
10,270 images.

For the media library the trade-off inverts. Reasoning is what makes the model
actually read the numbers, axis labels and captions off an infographic, and such
a collection is small enough that the price difference is noise ($0.13 versus
$0.03 for a few hundred files). So `--source media` defaults to reasoning on.

Both defaults are overridable per run: `--reasoning` / `--no-reasoning`.

## The folder name is a hint, not a fact

Archive folders carry information the pixels do not: "2017, 4-8 May. Trip to the
coast" states a date and an occasion. It is passed into the prompt, with an
explicit instruction to trust the image when the two disagree — folder names are
often approximate, and a model told to obey them will confidently mislabel the
odd photo that was filed in the wrong place.

The same folder convention is read again, mechanically, when indexing: the date
chain is EXIF, then a timestamp in the file name, then month and year words in
the folder path. On that archive the chain reached 89% of images; the remaining
11% have no date signal anywhere, and date queries cannot find them.

## Search is hybrid because pure semantics cannot find a name

The model is told not to name people and not to guess geography, so the
description contains neither — and cosine similarity therefore cannot answer "who
is in this photo" or "which country is this". But the archive already knows,
because a human confirmed faces and places in a photo manager. The indexer copies
those fields, and the query is matched against them directly, with a bonus added
to the semantic score.

Bonus sizes (person 1.0, location 0.7, date 0.9/0.6/0.3) are chosen relative to
the cosine range of roughly 0.3-0.6 so that an exact match always outranks a
purely semantic one, while the semantic remainder still orders results within the
matches. Date bonuses scale with precision on purpose: a bare year matches
hundreds of photos and must not flatten everything else.

Matching is prefix-based rather than a real stemmer (four shared leading
characters). For an archive browsed by eye, an occasional false positive costs a
glance; a morphology table for every supported language costs far more.

## Near-duplicates are text similarity, honestly labelled

There is no visual model here. Clusters are built from the *description*
embeddings, which works because frames of one burst get near-identical
descriptions — and fails when unrelated images are described alike. Comparison
stays inside a single folder, which both bounds the work (the sum of squares per
folder instead of the square of the whole archive) and avoids merging unrelated
folders with similar subjects.

Clustering is connected components, not pairs: a burst of five frames must come
out as one cluster even when its first and last frame no longer resemble each
other. The tool only reports; deleting is a human decision.

## Failure modes are logged, not thrown

A run touches thousands of files, so nothing aborts the pass. Provider errors,
malformed answers and content-filter refusals become log lines with a status, and
a second pass decides what is worth retrying. Two specifics worth knowing:

* A `200` response with `content: null` is a silent refusal by the provider's
  content filter, not a bug — it is logged and accepted, not worked around.
* A model that loops on self-correction can spill its thinking into the
  description: thousands of characters where a few hundred are normal. Such
  records are kept out of the index (they also break the embedder's cache) and
  can be re-tagged deliberately with `fix_script_glitches.py`.
