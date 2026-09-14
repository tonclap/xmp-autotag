"""Command line search over the index.

    python scripts/search.py "kids playing in the snow"
    python scripts/search.py "mountains in Austria" --top 10 --media

Runs fully offline: the embedding model is local and no API key is needed.
"""
import argparse
import sys

import search_core as sc


def _force_utf8_output():
    """Windows consoles often default to a legacy code page, which turns a
    perfectly good description into mojibake or a UnicodeEncodeError."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    _force_utf8_output()
    ap = argparse.ArgumentParser(description="Semantic search over a tagged photo archive")
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--media", action="store_true", help="include the media library")
    args = ap.parse_args()

    sources = sc.DEFAULT_SOURCES | {"media"} if args.media else sc.DEFAULT_SOURCES
    result = sc.search(args.query, top=args.top, sources=sources)
    if "error" in result:
        sys.exit(result["error"])
    if result["skippedCjk"]:
        print(
            f"(skipped {result['skippedCjk']} records with the known character-set glitch)",
            file=sys.stderr,
        )

    print(f'query: "{args.query}"\n')
    if not result["results"]:
        print(f"nothing found among {result['total']} indexed images")
        return
    for i, entry in enumerate(result["results"], 1):
        tag = ""
        if entry["matchType"]:
            tag = f"  [{entry['matchType']}: {entry['matchLabel']}]"
        if entry["source"] != "photos":
            tag += f"  ({entry['source']})"
        print(f"{i}. [{entry['score']:.3f}] {entry['path']}{tag}")
        print(f"   {entry['description']}")
        print(f"   keywords: {', '.join(entry['keywords'])}\n")


if __name__ == "__main__":
    main()
