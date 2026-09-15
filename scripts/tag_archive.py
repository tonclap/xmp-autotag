"""Tag a whole archive: walk it, describe every image, write XMP sidecars.

    python scripts/tag_archive.py                 # the photo archive
    python scripts/tag_archive.py --source media  # the media library
    python scripts/tag_archive.py --limit 20      # a cautious first run

Idempotent by design. A file counts as done when its sidecar already contains
dc:subject, so the run can be interrupted at any point — Ctrl+C, a closed
laptop, a dead network — and restarted; it picks up where it stopped and costs
nothing for what is already tagged. Every file is appended to a JSONL log
(path, status, cost, error) which retry_failed.py reads afterwards.

Requests are network-bound, so they run in a thread pool. Writes go to one
sidecar per image and never to a shared file, so the only shared state is the
log and the counters, both lock-protected.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Sequence

import requests

import config
import prompts
import vlm
import xmp


@dataclass
class Source:
    """One collection to walk, with the prompt and settings that fit it."""

    name: str
    root: Path
    prompt: Callable[[str], str]
    # Reasoning mode: off for photographs (no quality gain, several times the
    # price), on for charts and memes. See docs/DESIGN.md.
    reasoning: bool
    subdirs: Sequence[str] = field(default_factory=tuple)

    @property
    def log_path(self):
        return config.output_path(f"tag_{self.name}_log.jsonl")

    @property
    def backup_dir(self):
        return config.OUTPUT_DIR / f"xmp_backup_{self.name}"


def photos_source():
    root = config.require_photo_root()
    return Source(
        name="photos",
        root=root,
        prompt=lambda ctx: prompts.photo_prompt(ctx),
        reasoning=False,
    )


def media_source():
    if config.MEDIA_ROOT is None:
        sys.exit(
            "MEDIA_ROOT is not set - the media library is optional, see "
            ".env.example. Run without --source to tag the photo archive."
        )
    if not config.MEDIA_ROOT.is_dir():
        sys.exit(f"MEDIA_ROOT does not exist or is not a directory: {config.MEDIA_ROOT}")
    return Source(
        name="media",
        root=config.MEDIA_ROOT,
        prompt=lambda ctx: prompts.media_prompt(ctx),
        reasoning=True,
        subdirs=tuple(config.MEDIA_SUBDIRS),
    )


SOURCE_FACTORIES = {"photos": photos_source, "media": media_source}


def get_source(name):
    factory = SOURCE_FACTORIES.get(name)
    if factory is None:
        sys.exit(f"unknown source {name!r} (expected: {', '.join(SOURCE_FACTORIES)})")
    return factory()


def configured_sources():
    """Sources that this machine actually has configured, photos first.

    Used by the indexer, which should quietly skip a media library that was
    never set up instead of failing.
    """
    found = []
    if config.PHOTO_ROOT is not None and config.PHOTO_ROOT.is_dir():
        found.append(photos_source())
    if config.MEDIA_ROOT is not None and config.MEDIA_ROOT.is_dir():
        found.append(media_source())
    return found


def discover(root, subdirs=(), exts=None):
    """Yield image files under root (recursively), filtered by extension.

    With subdirs, only those direct children of root are walked — the usual
    shape of a media library, where the rest of the tree holds files that make
    no sense to describe. Missing subdirs are skipped silently.
    """
    exts = exts or config.IMAGE_EXTS
    roots = [Path(root) / sub for sub in subdirs] if subdirs else [Path(root)]
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                yield path


def already_done(xmp_path):
    """True when this sidecar already carries our keywords."""
    return xmp.has_subject(xmp_path)


class Run:
    """One pass over one source: counters, log file, worker body."""

    def __init__(self, source, reasoning=None, log_path=None):
        self.source = source
        self.reasoning = source.reasoning if reasoning is None else reasoning
        self.log_path = log_path or source.log_path
        self.log_lock = Lock()
        self.stats_lock = Lock()
        self.stats = {
            "created": 0,
            "merged": 0,
            "skipped_done": 0,
            "skipped_has_subject": 0,
            "error": 0,
            "cost": 0.0,
        }

    def log(self, entry):
        with self.log_lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def bump(self, key, cost=0.0):
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + 1
            self.stats["cost"] += cost

    def describe(self, img_path):
        """Model call for one image; returns the vlm.describe() result dict."""
        context = prompts.folder_context(img_path, self.source.root)
        prompt = self.source.prompt(context)
        try:
            result = vlm.describe(str(img_path), prompt, reasoning=self.reasoning)
        except requests.RequestException as exc:
            return {"error": f"request failed: {exc}"}, context
        return result, context

    def process_one(self, img_path):
        xmp_path = img_path.with_suffix(".xmp")
        if already_done(xmp_path):
            self.bump("skipped_done")
            return

        result, context = self.describe(img_path)
        if "error" in result:
            self.log({"path": str(img_path), "status": "error", "error": result["error"]})
            self.bump("error")
            return
        if result.get("text") is None:
            # HTTP 200 with content=null: a silent refusal by the provider's
            # content filter rather than a bug on our side. Log it and move on.
            self.log({"path": str(img_path), "status": "error",
                      "error": "provider returned null content (silent refusal?)"})
            self.bump("error")
            return

        description, keywords = xmp.parse_text(result["text"])
        if not description or not keywords:
            self.log({"path": str(img_path), "status": "error",
                      "error": "failed to parse description/keywords"})
            self.bump("error")
            return

        subject_xml, desc_xml = xmp.build_fields(description, keywords)
        cost = result.get("cost") or 0

        if xmp_path.exists():
            _backup, err = xmp.merge_existing(xmp_path, subject_xml, desc_xml)
            if err:
                status = "skipped_has_subject" if "already has" in err else "error"
                self.log({"path": str(img_path), "status": status, "error": err})
                self.bump(status)
                return
            status = "merged"
        else:
            xmp.create_new(xmp_path, subject_xml, desc_xml)
            status = "created"

        self.log({"path": str(img_path), "status": status, "context": context, "cost": cost})
        self.bump(status, cost)


def run_pool(items, worker, run, workers, report_every=50):
    """Run `worker` over `items` in a thread pool, printing progress.

    Ctrl+C stops the run: the queued work is cancelled explicitly. Leaving the
    pool to its default shutdown does the opposite — the worker threads drain
    the whole queue first, so an interrupted run of ten thousand images kept
    calling the paid API for hours after the interrupt.
    """
    t0 = time.time()
    done = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            done += 1
            exc = future.exception()
            if exc:
                run.log({"path": str(futures[future]), "status": "error",
                         "error": f"unhandled: {exc}"})
                run.bump("error")
            if done % report_every == 0 or done == len(items):
                elapsed = time.time() - t0
                print(
                    f"[{done}/{len(items)}] elapsed={elapsed:.0f}s "
                    f"created={run.stats['created']} merged={run.stats['merged']} "
                    f"skipped={run.stats['skipped_done'] + run.stats['skipped_has_subject']} "
                    f"error={run.stats['error']} cost=${run.stats['cost']:.4f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print(f"\ninterrupted after {done} files - nothing queued will be sent",
              flush=True)
        raise
    finally:
        pool.shutdown(wait=True)
    return time.time() - t0


def cli(entry_point):
    """Run a command line entry point, turning Ctrl+C into a quiet exit code.

    Shared by every script that tags: an interrupt is a normal way to stop a run
    here, so it should read as "stopped", not as a traceback.
    """
    try:
        entry_point()
    except KeyboardInterrupt:
        raise SystemExit(130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="photos", choices=sorted(SOURCE_FACTORIES))
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many untagged files (useful for a first run)")
    ap.add_argument("--reasoning", dest="reasoning", action="store_true", default=None,
                    help="force the model's reasoning mode on")
    ap.add_argument("--no-reasoning", dest="reasoning", action="store_false",
                    help="force the model's reasoning mode off")
    args = ap.parse_args()

    source = get_source(args.source)
    config.require_api_key()
    run = Run(source, reasoning=args.reasoning)
    xmp.BACKUP_DIR = source.backup_dir
    source.backup_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in discover(source.root, source.subdirs)]
    pending = [p for p in images if not already_done(p.with_suffix(".xmp"))]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{source.name}: {len(images)} images, {len(pending)} to tag")
    if not pending:
        return

    elapsed = run_pool(pending, run.process_one, run, args.workers)
    print(f"\nfinished in {elapsed:.0f}s ({elapsed / 3600:.2f}h)")
    print(json.dumps(run.stats, ensure_ascii=False, indent=2))
    print(f"log: {source.log_path}")


if __name__ == "__main__":
    cli(main)
