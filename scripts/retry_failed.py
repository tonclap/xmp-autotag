"""Second pass: retry what tag_archive.py could not finish.

    python scripts/retry_failed.py [--source photos|media]

Reads the run log of a previous pass and takes another shot at everything that
came back as an error. Worth doing because most failures are transient or
size-related rather than final:

* HTTP 429 (rate limited) — retried with a growing delay.
* HTTP 413 (too large) — the uploader downscales oversized images, so files that
  failed before a resize threshold change may now pass.
* a content-filter refusal — retried once, then accepted as permanent. This is
  the provider's policy, not a bug here, and it is not worked around.
* a parse failure — retried once; models occasionally answer in the wrong shape.

Runs with fewer workers than the first pass: this pass exists because the
provider was pushing back, so it should not push equally hard again.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

import config
import prompts
import tag_archive
import vlm
import xmp

DEFAULT_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 3


def failed_paths(log_path):
    """Unique paths that ended in an error in a previous run, in log order."""
    if not log_path.exists():
        sys.exit(f"no run log at {log_path} - run tag_archive.py first")
    seen = set()
    out = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line of an interrupted run
            if rec.get("status") == "error" and rec.get("path") not in seen:
                seen.add(rec["path"])
                out.append(Path(rec["path"]))
    return out


class RetryRun(tag_archive.Run):
    """Same worker body, but the model call retries before giving up."""

    def describe(self, img_path):
        context = prompts.folder_context(img_path, self.source.root)
        prompt = self.source.prompt(context)
        last_err = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                result = vlm.describe(str(img_path), prompt, reasoning=self.reasoning)
            except requests.RequestException as exc:
                result = {"error": f"request failed: {exc}"}
            if "error" not in result:
                return result, context
            last_err = result["error"]
            transient = last_err.startswith("HTTP 429") or "data_inspection_failed" in last_err
            if not transient:
                break  # 400/413/parse failures do not get better by repeating
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
        return {"error": last_err}, context


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="photos", choices=sorted(tag_archive.SOURCE_FACTORIES))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    source = tag_archive.get_source(args.source)
    config.require_api_key()
    targets = [p for p in failed_paths(source.log_path) if not tag_archive.already_done(p.with_suffix(".xmp"))]
    print(f"{source.name}: {len(targets)} files to retry")
    if not targets:
        return

    run = RetryRun(source)
    xmp.BACKUP_DIR = source.backup_dir
    source.backup_dir.mkdir(parents=True, exist_ok=True)

    elapsed = tag_archive.run_pool(targets, run.process_one, run, args.workers)
    print(f"\nfinished in {elapsed:.0f}s")
    print(json.dumps(run.stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    tag_archive.cli(main)
