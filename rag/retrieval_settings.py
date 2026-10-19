"""Retrieval configuration read from environment variables."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

RETRIEVAL_MODE_VARIABLE = "RETRIEVAL_MODE"
QDRANT_URL_VARIABLE = "QDRANT_URL"


class RetrievalMode(StrEnum):
    TFIDF = "tfidf"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


DEFAULT_RETRIEVAL_MODE = RetrievalMode.HYBRID


class InvalidRetrievalModeError(ValueError):
    """Raised when RETRIEVAL_MODE names a mode that does not exist."""


@dataclass(frozen=True)
class RetrievalSettings:
    mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE
    qdrant_url: str | None = None


def load_retrieval_settings(environ: Mapping[str, str]) -> RetrievalSettings:
    return RetrievalSettings(
        mode=_parse_mode(environ.get(RETRIEVAL_MODE_VARIABLE, "")),
        qdrant_url=environ.get(QDRANT_URL_VARIABLE, "").strip() or None,
    )


def _parse_mode(raw_mode: str) -> RetrievalMode:
    normalized_mode = raw_mode.strip().lower()
    if not normalized_mode:
        return DEFAULT_RETRIEVAL_MODE
    try:
        return RetrievalMode(normalized_mode)
    except ValueError:
        valid_modes = ", ".join(mode.value for mode in RetrievalMode)
        raise InvalidRetrievalModeError(
            f"{RETRIEVAL_MODE_VARIABLE} must be one of: {valid_modes} (got {raw_mode!r})"
        ) from None
