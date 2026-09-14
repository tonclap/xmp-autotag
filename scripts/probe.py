"""Dry run: see what the model says about a handful of images, write nothing.

    python scripts/probe.py                          # 10 images spread over the archive
    python scripts/probe.py --limit 20 --source media
    python scripts/probe.py path/to/one.jpg path/to/another.jpg

This is the honest first step before tagging thousands of files: it calls the
model, prints the answers and saves them to output/, but never touches a
sidecar. Use it to check the output language, the level of detail, and whether
reasoning mode is worth its price on your material.
"""
import argparse
import json
import time
from pathlib import Path

import requests

import config
import prompts
import tag_archive
import vlm


def sample(source, limit):
    """`limit` images spread evenly across the archive, not just the first ones.

    Taking the first N files means sampling one folder, which says nothing about
    an archive spanning decades.
    """
    files = list(tag_archive.discover(source.root, source.subdirs))
    if not files or limit >= len(files):
        return files
    step = len(files) / limit
    return [files[int(i * step)] for i in range(limit)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path, help="specific images (default: a sample)")
    ap.add_argument("--source", default="photos", choices=sorted(tag_archive.SOURCE_FACTORIES))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--reasoning", dest="reasoning", action="store_true", default=None)
    ap.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    args = ap.parse_args()

    source = tag_archive.get_source(args.source)
    config.require_api_key()
    reasoning = source.reasoning if args.reasoning is None else args.reasoning
    images = args.paths or sample(source, args.limit)
    if not images:
        print(f"no images found under {source.root}")
        return

    results = []
    out_json = config.output_path(f"probe_{source.name}.json")
    for i, path in enumerate(images, 1):
        if not Path(path).exists():
            print(f"[{i}/{len(images)}] MISSING: {path}")
            results.append({"path": str(path), "error": "file not found"})
            continue
        context = prompts.folder_context(path, source.root)
        prompt = source.prompt(context)
        print(f"[{i}/{len(images)}] {path}  (context: {context})")
        try:
            result = vlm.describe(str(path), prompt, reasoning=reasoning)
        except requests.RequestException as exc:
            result = {"error": f"request failed: {exc}"}
        result["path"] = str(path)
        result["context"] = context
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            text = result.get("text") or ""
            print("  " + text.replace("\n", "\n  "))
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(1)  # be a polite API client on a sample this small

    total_cost = sum(r.get("cost") or 0 for r in results)
    print(f"\nsaved: {out_json}")
    print(f"total cost: ${total_cost:.4f} for {len(results)} images "
          f"(reasoning={'on' if reasoning else 'off'})")


if __name__ == "__main__":
    main()
