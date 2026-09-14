# Known issues

Properties of the result, not a bug list waiting to be fixed. Each one was seen on
a real run of roughly 10,000 images and accepted deliberately; the reasoning is
part of the entry.

**Near-duplicate clusters measure text, not pixels.** Frames described in similar
words end up in one cluster even when they are visually different — several
different subjects photographed at the same spot in the same style is the typical
case. Mitigation by design: the tool only shows clusters, it never deletes.

**The person and place matcher fires on shared prefixes.** Four leading
characters in common count as a match, so a query word can occasionally hit an
unrelated name or town. Low impact for browsing by eye; the alternative is a
per-language morphology table.

**A model can answer in the wrong script.** On one full run with reasoning
disabled, about 19% of images came back with stray CJK characters mixed into
otherwise clean text. Re-running those with reasoning enabled fixed essentially
all of them (`scripts/fix_script_glitches.py`); the same run with reasoning on
from the start produced none. Records still carrying the glitch are excluded from
the index and counted, not silently dropped.

**A model can loop and spill its reasoning into the description.** Thousands of
characters instead of a few hundred. `MAX_DESCRIPTION_CHARS` keeps such records
out of the index — they also break the ONNX embedder, whose cache is not built for
that sequence length — but the damaged sidecar on disk is only repaired when you
ask for it.

**Some images will never get tags.** Provider content filters refuse a small
number of images with `200 OK` and an empty body. This is a policy decision on
their side and is not worked around here; those files are logged as permanent
errors.

**Images sharing a basename share a sidecar.** `photo.jpeg` and `photo.png` in one
folder both map to `photo.xmp` — a property of the sidecar convention, not of this
pipeline. In a parallel run, whichever finishes first gets its description stored.

**Date coverage is partial.** The chain EXIF > file name > folder name reached 89%
of images on the reference archive. The rest have no date signal at all, and date
queries cannot reach them.

**RAW support is limited to the embedded preview.** Pillow does not decode RAW, so
oversized RAW files are uploaded via their embedded JPEG preview. A RAW file
without such a preview and above the request size limit stays untagged.

**Sidecars occasionally vanished after a successful write.** Two files out of
10,270 reported `created` and had no sidecar on a later check; re-creating one of
them did not stick on the first attempt. The cause was never established and was
not pursued for two files. Worth watching if you see it at a higher rate.
