"""Shared, read-only course model and build-time checks."""

import argparse
import os
import re
import tempfile
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from listings import expand
from lxml import etree as ET
from matrix import NAME, snapshot
from source_highlight import highlight

GENERATED = {}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "course.xml")
    parser.add_argument("--output", type=Path, default=ROOT)
    return parser.parse_args()


def text(node):
    return "".join(node.itertext()).strip()


def topics(chapter):
    return chapter.findall('section[@class="topic"]')


def chapter_url(chapter):
    slug = chapter.get("slug")
    return "roadmap.html" if slug.startswith("00-") else "lessons/" + slug + ".html"


RAW_MATH = re.compile(
    r"\\(?:\(|\[|frac\b|sqrt\b|begin\{|[a-zA-Z]+\{)|\$\$|"
    r"(?<![A-Za-z0-9_./])[A-Za-zα-ω]_(?:[A-Za-z0-9]+|\{[^}]+\})(?![A-Za-z0-9_./])"
)


def check_formulas(chapter):
    for section in chapter.iter("section"):
        for equation in section.xpath('./p[@class="eq"]'):
            if not equation.xpath('.//*[local-name()="math"]'):
                raise ValueError(f"Missing equation in {section.get('id')}")
            if re.search(r"\[[A-Za-z0-9]+×[A-Za-z0-9]+\]", text(equation)):
                raise ValueError(f"Unattached matrix dimensions in {section.get('id')}")
        for node in section.xpath(
            './/p[@class="prose"]//text() | .//td//text() | .//figcaption//text()'
        ):
            parent = node.getparent()
            if node.is_tail:
                parent = parent.getparent()
            if any(
                ET.QName(el).localname in ("math", "code", "pre", "listing")
                or el.get("class") in ("inline-id", "code")
                for el in (parent, *parent.iterancestors())
            ):
                continue
            match = RAW_MATH.search(node)
            if match:
                raise ValueError(f"Unrendered {match[0]!r} in {section.get('id')}")


def read(path):
    chapters = (
        ET.parse(path, ET.XMLParser(resolve_entities=False, no_network=True))
        .getroot()
        .findall("chapter")
    )
    GENERATED.clear()
    digest.cache_clear()
    seen = set()
    all_ids = set()
    for ch in chapters:
        if (
            not re.fullmatch(r"\d\d-[a-z0-9-]+", ch.get("slug", ""))
            or ch.get("slug") in seen
        ):
            raise ValueError("Invalid or duplicate chapter slug")
        seen.add(ch.get("slug"))
        ids = [e.get("id") for e in ch.iter() if e.get("id")]
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError("Duplicate id in " + ch.get("slug"))
        all_ids.update(ids)
        local_ids = set(ids)
        check_formulas(ch)
        broken = {
            link[1:]
            for link in ch.xpath(".//a/@href")
            if link.startswith("#") and link[1:] not in local_ids
        }
        if broken:
            raise ValueError(
                f"Unknown link targets in {ch.get('slug')}: {sorted(broken)}"
            )
        for math in ch.xpath('.//*[local-name()="math"]'):
            if ET.QName(math).namespace != "http://www.w3.org/1998/Math/MathML":
                raise ValueError(
                    "Formula missing the MathML namespace in " + ch.get("slug")
                )
        for lab in ch.xpath(".//*[@data-matrix-lab]"):
            if NAME in GENERATED:
                raise ValueError("The course supports one four-position attention lab")
            GENERATED[NAME] = snapshot(lab)
        for sec in topics(ch):
            if sec[0].tag not in ("h2", "h3") or not sec.get("id"):
                raise ValueError("A topic needs an id and a heading")
        question_ids = {q.get("id") for q in ch.iter("exercise")}
        for q in ch.iter("exercise"):
            if q.get("next") and q.get("next") not in question_ids:
                raise ValueError("Unknown next exercise for " + str(q.get("id")))
            if q.find("p[@id='prompt-" + str(q.get("id")) + "']") is None:
                raise ValueError(
                    "Expected a question paragraph with id prompt-" + str(q.get("id"))
                )
            if not q.get("id") or not any(text(p) for p in q.findall("p")):
                raise ValueError("Exercise needs an id and a prompt")
            if (
                not q.get("title")
                or q.find("answer") is None
                or not q.find("answer").get("id")
                or not text(q.find("answer"))
            ):
                raise ValueError("Incomplete exercise " + str(q.get("id")))
        for listing in list(ch.iter("listing")):
            expand(listing)
        for fig in ch.xpath('.//figure[@class="code"]'):
            if len(fig.xpath('.//span[@class="cl"]')) < 5:
                raise ValueError(
                    "Code sample shorter than five lines: " + text(fig)[:60]
                )
            highlight(fig)
        for im in ch.iter("img"):
            image_path(im.get("src"))
    if not chapters or not chapters[0].get("slug").startswith("00-"):
        raise ValueError("Expected the roadmap first")
    return chapters


def image_path(url):
    p = (ROOT / url.split("?")[0]).resolve()
    if not p.is_relative_to((ROOT / "assets/figures").resolve()) or not p.is_file():
        raise ValueError("Missing or invalid course image: " + url)
    return p


@lru_cache(None)
def digest(path):
    return sha256(image_bytes(path)).hexdigest()[:12]


def image_bytes(path):
    if path.parent == ROOT / "assets/figures" and path.name in GENERATED:
        return GENERATED[path.name]
    return path.read_bytes()


def write(path, data):
    if isinstance(data, str):
        data = data.encode("utf8")
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    try:
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
