"""Attach section-scoped symbol notes to MathML without altering the equations."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from html import escape
import json
from itertools import groupby
from pathlib import Path
import re

from lxml import etree as ET

from equations import MATH


@lru_cache
def catalog():
    groups = defaultdict(dict)
    for line in Path(__file__).with_name("symbol_notes.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            scopes = line.strip("[]").split()
            continue
        aliases, note = line.split(" | ", 1)
        keys, label = aliases.split(" / ", 1)
        for scope in scopes:
            for key in keys.split():
                groups[scope][key] = (label, note)
    return groups


def compact(element):
    return re.sub(r"\s+", "", "".join(element.itertext()))


def symbol_key(node):
    if node.get("data-symbol"):
        return node.get("data-symbol")
    key = compact(node)
    if node.get("mathvariant") == "double-struck":
        return {"R": "ℝ", "N": "ℕ", "Z": "ℤ"}.get(key, key)
    return key


def symbols(element, known):
    """Treat a scripted name as one symbol, and a dimension label as notation."""
    tag = ET.QName(element).localname
    if tag == "munder" and len(element) == 2 and "⏟" in compact(element[1]):
        yield from symbols(element[0], known)
        return
    if tag in ("msub", "msup", "msubsup", "mover"):
        key = symbol_key(element)
        if key in known or (tag != "msup" and len(key) <= 14):
            yield key
            return
        yield from symbols(element[0], known)
        if tag == "msup" and compact(element[-1]) not in ("⊤", "T"):
            yield from symbols(element[-1], known)
        return
    if tag in ("mi", "mtext"):
        key = symbol_key(element)
        if key in known or (element.get("mathvariant") != "normal" and re.fullmatch(r"[A-Za-z\u0370-\u03ffℓ][A-Za-z\u0370-\u03ffℓ0-9_]*", key)):
            yield key
        return
    if element.get("mathvariant") == "normal":
        return
    # Earlier source material spells identifiers (such as blockDim) in upright
    # MathML letters. Never interpret their constituent letters as dimensions.
    def upright(child):
        return child.tag == MATH + "mi" and child.get("mathvariant") == "normal"
    for grouped, children in groupby(element, upright):
        if grouped:
            word = "".join(compact(child) for child in children)
            if word in known or (word and word not in {"ms","ns","s","byte","bytes","GiB","MiB","KiB","GB","MB","KB",
                                     "FLOP","FLOPs","GFLOPs","exp","sin","cos","ln","log","max","min","mod"}):
                yield word
        else:
            for child in children:
                yield from symbols(child, known)


def occurrence_context(section):
    """Index only the sentence actually containing each occurrence, never nearby text."""
    formulas = []
    sentences = {}
    for block in section.xpath(".//p[not(@class='eq')] | .//dd | .//figcaption | .//td | .//th | .//li"):
        if block.find(".//" + MATH + "math") is None:
            continue

        def inline(node):
            if node.tag == MATH + "math":
                mark = chr(0xf0000 + len(formulas))
                formulas.append(node)
                return mark
            return (node.text or "") + "".join(inline(n) + (n.tail or "") for n in node)

        for sentence in re.split(r"(?<=[。！？；])", inline(block)):
            sentence = sentence.strip()
            if len(re.findall(r"[\u4e00-\u9fff]", sentence)) < 5:
                continue
            for mark in sentence:
                if ord(mark) >= 0xf0000:
                    sentences[formulas[ord(mark)-0xf0000]] = sentence
    return sentences, formulas


def render_sentence(sentence, formulas):
    # Only generated MathML can supply markup. Prose, code names and links become plain text.
    parts = []
    for fragment in re.split(r"([\U000f0000-\U000ffffd])", sentence):
        if len(fragment) == 1 and ord(fragment) >= 0xf0000:
            math = formulas[ord(fragment)-0xf0000]
            parts.append(ET.tostring(math, encoding="unicode", with_tail=False))
        else:
            parts.append(escape(fragment))
    return "".join(parts)


def typeset_label(node):
    math = ET.Element(MATH + "math", nsmap={None: MATH[1:-1]})
    math.append(deepcopy(node))
    return ET.tostring(math, encoding="unicode", with_tail=False)


def prepare_symbol_notes(chapter):
    for section in chapter.xpath("./section"):
        sid = section.get("id", "")
        known = {}
        for scope in ("*", sid.split("-")[0] + "-", sid):
            known.update(catalog().get(scope, {}))
        spans = section.xpath('.//span[@class="formula"][not(ancestor::a or ancestor::button or ancestor::pre or ancestor::code)]')
        if not spans:
            continue
        sentences, formulas = occurrence_context(section)
        notes = {}
        base = known
        exercise_scopes = {}
        worked = ""
        for block in section:
            if block.get("id", "").startswith("handcalc-"):
                worked = block.get("id")
            elif block.tag in ("exercise","listing") or block.get("class") == "reading":
                worked = ""
            if worked:
                exercise_scopes.update((node, worked) for node in block.iter("span") if node.get("class") == "formula")
        for span in spans:
            known = base | catalog().get(exercise_scopes.get(span), {})
            math = span.find(MATH + "math")
            if math is None:
                continue
            keys = list(dict.fromkeys(symbols(math, known)))
            if not keys and span.get("data-note") is None:
                continue
            selected = []
            if "data-note" in span.attrib:
                selected.append((span.attrib.pop("data-note-title", "本式的计算"), span.attrib.pop("data-note")))
            labels = {}
            for node in math.iter():
                if ET.QName(node).localname in {"mi","mtext","msub","msup","msubsup","mover"}:
                    labels.setdefault(symbol_key(node), node)
            for key in keys:
                if key in known:
                    note = known[key]
                    if key in labels:
                        note = (*note, "text", typeset_label(labels[key]))
                    if note not in selected:
                        selected.append(note)
            if not selected and math in sentences:
                selected.append(("正文在此处的说明", render_sentence(sentences[math], formulas), "mathml"))
            if not selected:
                continue
            ids = [notes.setdefault(note, len(notes)) for note in selected]
            span.attrib.update({"data-symbols": " ".join(map(str, ids)), "role": "button", "tabindex": "0",
                                "aria-haspopup": "dialog", "aria-expanded": "false", "title": "查看本式的符号说明"})
        if notes:
            template = ET.SubElement(section, "template", {"class":"symbol-notes"})
            template.text = json.dumps(list(notes), ensure_ascii=False, separators=(",", ":"))
