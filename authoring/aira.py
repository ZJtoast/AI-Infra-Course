"""Embed the current page's prewritten reading cards and local pet assets."""

from hashlib import sha256
from functools import lru_cache
from html import escape
import json
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "aira"


@lru_cache(maxsize=1)
def artwork_version():
    revision = sha256()
    for path in sorted(ASSETS.rglob("*.webp")):
        if path.name.endswith(".building.webp"):
            continue
        revision.update(path.relative_to(ASSETS).as_posix().encode())
        revision.update(sha256(path.read_bytes()).digest())
    return revision.hexdigest()[:12]


def reading_companion(page, prefix, cards):
    if not cards:
        raise ValueError("Aira needs at least one reading card")
    for key, card in cards.items():
        if not all(isinstance(card.get(field), str) and card[field].strip()
                   for field in ("fact", "question", "answer")):
            raise ValueError(f"Incomplete reading card: {key}")

    def url(name):
        revision = sha256((ASSETS / name).read_bytes()).hexdigest()[:12]
        return escape(prefix + "assets/aira/" + name + "?v=" + revision, quote=True)

    payload = json.dumps(cards, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return page.replace("</head>", f'<link rel="stylesheet" href="{url("aira.css")}"></head>', 1).replace(
        "</body>", f'<script type="application/json" id="aira-cards">{payload}</script>'
        f'<script defer src="{url("aira.js")}" data-art-version="{artwork_version()}"></script></body>', 1
    )
