"""Name normalization for the assertion boundary.

Pure string canonicalization with no sema dependencies so it can be imported
from the models layer without creating an import cycle.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_name(text: str) -> str:
    """Canonicalize a name/alias so cosmetic variants map to one graph node.

    Applies unicode NFC, collapses internal whitespace runs to a single
    space, and strips surrounding whitespace. Case is preserved.
    """
    nfc = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RUN.sub(" ", nfc).strip()
