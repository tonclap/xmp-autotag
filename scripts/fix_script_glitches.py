"""Re-tag images whose stored text came out in the wrong script.

    python scripts/fix_script_glitches.py [--source photos|media]

Symptom, observed on a five-figure run: a share of the images come back with
stray CJK characters mixed into an otherwise clean description or keyword list.
It is a language-switching glitch in the model, and it correlates with reasoning
being disabled — a sample run with reasoning on produced none of it.

So the repair is: find the affected records in the index, ask again with
reasoning enabled and an explicit "one language only" instruction, and replace
just dc:subject and dc:description in the sidecar. If the answer still contains
the glitch, the record is logged as such and the file is left alone — writing
another broken description over a broken one helps nobody.

Resumable: every processed path is appended to a log that the next run skips.
Note that being in the log means "attempted", not "clean" — when you delete
damaged records to try again, delete their log lines too, or the retry will
silently skip them.
"""
import argparse
import json
import sys
from pathlib import Path

import config
import search_core as sc
import tag_archive
import xmp

# The same pattern the index reader uses to exclude these records, imported
# rather than restated: two copies of "what counts as the glitch" would let the
# repair pass and the index disagree about which files are damaged.
CJK_RE = sc.CJK_RE


def affected_paths(index_path):
    """Indexed records whose text contains characters it should not."""
    if not index_path.exists():
        sys.exit(f"no index at {index_path} - run build_search_index.py first")
    found = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("description", "") + " ".join(record.get("keywords") or [])
            if CJK_RE.search(text):
                found.append(record["path"])
    return found


def already_attempted(log_path):
    done = set()
    if not log_path.exists():
        return done
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


class FixRun(tag_archive.Run):
    """Re-ask with reasoning on, then replace the existing fields in place."""

    def process_one(self, img_path):
        img_path = Path(img_path)
        xmp_path = img_path.with_suffix(".xmp")
        result, _context = self.describe(img_path)

        if "error" in result or result.get("text") is None:
            error = result.get("error") or "provider returned null content (silent refusal?)"
            self.log({"path": str(img_path), "status": "error", "error": error})
            self.bump("error")
            return

        description, keywords = xmp.parse_text(result["text"])
        if not description or not keywords:
            self.log({"path": str(img_path), "status": "error", "error": "failed to parse"})
            self.bump("error")
            return

        if CJK_RE.search(description + " ".join(keywords)):
            self.log({"path": str(img_path), "status": "still_glitched",
                      "description": description, "keywords": keywords})
            self.bump("still_glitched")
            return

        subject_xml, desc_xml = xmp.build_fields(description, keywords)
        ok, err = xmp.replace_fields(xmp_path, subject_xml, desc_xml)
        cost = result.get("cost") or 0
        entry = {
            "path": str(img_path),
            "status": "fixed" if ok else "xmp_error",
            "description": description,
            "keywords": keywords,
            "cost": cost,
        }
        if err:
            entry["error"] = err
        self.log(entry)
        self.bump("fixed" if ok else "xmp_error", cost)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="photos", choices=sorted(tag_archive.SOURCE_FACTORIES))
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    args = ap.parse_args()

    source = tag_archive.get_source(args.source)
    config.require_api_key()
    log_path = config.output_path(f"fix_glitches_{source.name}_log.jsonl")

    affected = affected_paths(sc.INDEX_PATH)
    attempted = already_attempted(log_path)
    remaining = [p for p in affected if p not in attempted]
    print(f"glitched records: {len(affected)}, already attempted: {len(attempted)}, "
          f"to re-tag: {len(remaining)}")
    if not remaining:
        return

    # Reasoning on (that is the whole point of this pass) and its own log file,
    # so the tagging log keeps meaning "what the first pass did".
    run = FixRun(source, reasoning=True, log_path=log_path)
    xmp.BACKUP_DIR = source.backup_dir
    source.backup_dir.mkdir(parents=True, exist_ok=True)

    elapsed = tag_archive.run_pool(remaining, run.process_one, run, args.workers)
    print(f"\nfinished in {elapsed:.0f}s")
    print(json.dumps(run.stats, ensure_ascii=False, indent=2))
    print(f"log: {log_path}")


if __name__ == "__main__":
    tag_archive.cli(main)
