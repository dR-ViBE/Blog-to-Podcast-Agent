"""
ingestion/ingest.py

PURPOSE:
    CLI entrypoint for the multi-source ingestion pipeline.
    Replaces the old hardcoded ingestion.py.

USAGE:
    # Ingest a URL
    poetry run python -m ingestion.ingest --source "https://lilianweng.github.io/posts/..."

    # Ingest a PDF
    poetry run python -m ingestion.ingest --source "./docs/paper.pdf"

    # Ingest a plain text file
    poetry run python -m ingestion.ingest --source "./docs/notes.txt"

    # Ingest all supported files in a directory
    poetry run python -m ingestion.ingest --source "./docs/"

    # Ingest multiple sources at once
    poetry run python -m ingestion.ingest \\
        --source "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/" \\
        --source "./docs/extra_notes.pdf"

    # Show help
    poetry run python -m ingestion.ingest --help

WHY A CLI:
    Separating ingestion from the runtime API is a production best practice.
    Ingestion is an offline, admin-level task. It should never be triggered
    by a user request to the API — it's run once (or on a schedule) by a
    data engineer or a CI/CD pipeline step.
"""

import argparse
import sys
from ingestion.loaders import ingest_source


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ingestion.ingest",
        description=(
            "Blog-to-Podcast Agent — Multi-Source Ingestion CLI\n\n"
            "Ingests content from a URL, PDF, text file, or directory into ChromaDB.\n"
            "Supported formats: .pdf, .txt, .md, .rst, and any HTTP/HTTPS URL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m ingestion.ingest --source https://example.com/post\n"
            "  python -m ingestion.ingest --source ./docs/paper.pdf\n"
            "  python -m ingestion.ingest --source ./docs/\n"
            "  python -m ingestion.ingest --source url1 --source ./file.pdf\n"
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        required=True,
        metavar="SOURCE",
        help=(
            "Source to ingest. Can be a URL, a file path (.pdf/.txt/.md/.rst), "
            "or a directory. Repeat --source to ingest multiple sources."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    total_chunks = 0
    failed = []

    for source in args.sources:
        try:
            chunks = ingest_source(source)
            total_chunks += chunks
        except Exception as e:
            print(f"\n  ❌ Failed to ingest '{source}': {e}")
            failed.append(source)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Sources processed : {len(args.sources)}")
    print(f"  Successful        : {len(args.sources) - len(failed)}")
    print(f"  Failed            : {len(failed)}")
    print(f"  Total chunks      : {total_chunks}")
    if failed:
        print(f"\n  Failed sources:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}\n")

    # Exit with non-zero code if any source failed (useful for CI)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
