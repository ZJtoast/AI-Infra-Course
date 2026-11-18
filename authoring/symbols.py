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


def symbols(element, known):
    """Treat a scripted name as one symbol, and a dimension label as notation."""
    tag = ET.QName(element).localname
    if tag == "munder" and len(element) == 2 and "⏟" in compact(element[1]):
        yield from symbols(element[0], known)
        return
    if tag in ("msub", "msup", "msubsup", "mover"):
        key = compact(element)
        if key in known or (tag != "msup" and len(key) <= 14):
            yield key
            return
        yield from symbols(element[0], known)
        if tag == "msup" and compact(element[-1]) not in ("⊤", "T"):
            yield from symbols(element[-1], known)
        return
    if tag in ("mi", "mtext"):
        key = compact(element)
        if element.get("mathvariant") != "normal" and re.fullmatch(r"[A-Za-z\u0370-\u03ffℓ][A-Za-z\u0370-\u03ffℓ0-9_]*", key):
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


def exercise_scope(element):
    return next((n for n in element.iterancestors() if n.tag in ("answer", "exercise")), None)


def collect_sentences(section, known, order):
    # Keep MathML placeholders intact when splitting an explanatory sentence.
    formulas = []
    passages = []
    blocks = section.xpath(".//p | .//dd | .//figcaption | .//td | .//li")
    for block in blocks:
        if block.get("class") == "eq" or block.xpath("ancestor::pre | ancestor::figure[@class='code']"):
            continue
        if not any("\u4e00" <= c <= "\u9fff" for c in "".join(block.itertext())):
            continue
        def inline(node):
            if node.tag == MATH + "math":
                idx = len(formulas)
                formulas.append(node)
                return chr(0xf0000 + idx)
            return (node.text or "") + "".join(inline(n) + (n.tail or "") for n in node)
        for sentence in re.split(r"(?<=[。！？；])", inline(block)):
            sentence = sentence.strip()
            if not 8 <= len(re.findall(r"[\u4e00-\u9fff]", sentence)) or len(sentence) > 440:
                continue
            keys = set(re.findall(r"(?<![\w.])[A-Za-zΑ-ω]+(?![a-zA-Z0-9_.])", sentence, re.ASCII))
            for mark in sentence:
                if ord(mark) >= 0xf0000:
                    keys.update(symbols(formulas[ord(mark)-0xf0000], known))
            passages.append((order[block], exercise_scope(block), keys, sentence))
    return passages, formulas


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
        order = {node: i for i, node in enumerate(section.iter()) if not str(node.tag).startswith(MATH)}
        passages, formulas = collect_sentences(section, known, order)
        notes = {}
        for span in spans:
            math = span.find(MATH + "math")
            if math is None:
                continue
            keys = list(dict.fromkeys(symbols(math, known)))
            if not keys:
                continue
            selected = []
            labels = {}
            for node in math.iter():
                if ET.QName(node).localname in {"mi","mtext","msub","msup","msubsup","mover"}:
                    labels.setdefault(compact(node), node)
            for key in keys:
                if key in known:
                    note = known[key]
                    if key in labels:
                        note = (*note, "text", typeset_label(labels[key]))
                    if note not in selected:
                        selected.append(note)
            unknown = set(keys) - known.keys()
            if unknown:
                here = order[span]
                scope = exercise_scope(span)
                parent = span.getparent()
                while parent is not section and parent.tag not in ("p", "td", "dd", "li", "figcaption"):
                    parent = parent.getparent()
                owner = order[parent]
                choices = []
                for pos, zone, words, sentence in passages:
                    if zone is not None and zone is not scope:
                        continue
                    hits = unknown & words
                    distance = abs(pos-here)
                    if not hits and (selected or (pos != owner and distance > 150)):
                        continue
                    same = pos == owner
                    definition = bool(re.search("表示|代表|称为|记为|记作|为.+[维行列]|每[条个组头行列]|宽度|概率|分母|对应|索引|取值|等于|决定", sentence))
                    score = (bool(hits), len(hits), same, definition and distance < 120, -distance)
                    choices.append((score, sentence))
                if choices:
                    _, sentence = max(choices, key=lambda item: item[0])
                    selected.append(("本节中的用法", render_sentence(sentence, formulas), "mathml"))
            if not selected:
                continue
            ids = [notes.setdefault(note, len(notes)) for note in selected]
            span.attrib.update({"data-symbols": " ".join(map(str, ids)), "role": "button", "tabindex": "0",
                                "aria-haspopup": "dialog", "aria-expanded": "false", "title": "查看本式的符号说明"})
        if notes:
            template = ET.SubElement(section, "template", {"class":"symbol-notes"})
            template.text = json.dumps(list(notes), ensure_ascii=False, separators=(",", ":"))
