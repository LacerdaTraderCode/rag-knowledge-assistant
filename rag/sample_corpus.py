"""Loads the small bundled sample knowledge base, used by the demo app and the tests."""

from pathlib import Path

SAMPLE_CORPUS_DIR = Path(__file__).resolve().parent.parent / "docs_sample"


def load_sample_documents() -> dict[str, str]:
    return {path.stem: path.read_text() for path in sorted(SAMPLE_CORPUS_DIR.glob("*.md"))}
