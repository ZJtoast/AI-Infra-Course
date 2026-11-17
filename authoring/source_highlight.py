"""Lex each complete excerpt once and share the resulting styles between editions."""

import re
from functools import lru_cache
from typing import ClassVar

from listings import language_of
from lxml import etree as ET
from pygments.lexer import RegexLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import (
    Comment,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)
from pygments.util import ClassNotFound

# Screen and Word use dark code panels; browsers print against a light background.
COLORS = {
    "keyword": ("CDB1FF", "5E258F"),
    "type": ("82E0D1", "005C55"),
    "function": ("F6D78E", "714900"),
    "string": ("F6B797", "85380D"),
    "comment": ("A2B9A0", "3C6143"),
    "number": ("B6DEA3", "345E26"),
    "variable": ("9AD8FF", "154F7C"),
    "operator": ("E5B2E9", "733B79"),
    "added": ("98DDA4", "1D6234"),
    "removed": ("FF9F9F", "972F38"),
}


class SassLexer(RegexLexer):
    """NVIDIA disassembly, including instruction offsets, predicates and registers."""

    tokens: ClassVar[dict] = {
        "root": [
            (r"/\*[\s\S]*?\*/|//[^\n]*", Comment),
            (r"@!?U?P(?:T|\d+)\b", Keyword.Reserved),
            (r"\b(?:U?R(?:Z|\d+)|U?P(?:T|\d+)|SR_\w+)\b", Name.Variable),
            (r"\b[A-Z][A-Z0-9]*(?:\.[A-Za-z0-9]+)*", Keyword),
            (r"0[xX][\da-fA-F]+|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", Number),
            (r"\w+|\s+", Text),
            (r".", Punctuation),
        ]
    }


class PseudocodeLexer(RegexLexer):
    tokens: ClassVar[dict] = {
        "root": [
            (r"//[^\n]*|#[^\n]*", Comment.Single),
            (
                r"\b(?:for|each|in|if|else|while|return|when|repeat|until|then|otherwise|parallel|continue|break)\b",
                Keyword,
            ),
            (r"\b(?:producer|consumer)(?=:)", Name.Label),
            (
                r"(?m)^(?:choose|bound|quantize|issue|reset|restore|wait|compute|load|store|finish|release|drain|pack|center|encode)\b",
                Name.Function,
            ),
            (r"[A-Za-z_]\w*(?=\s*\()", Name.Function),
            (r"\d+(?:\.\d+)?|∞", Number),
            (r"[←×÷−@+*/<>=]", Operator),
            (r"[\w]+|\s+", Text),
            (r".", Punctuation),
        ]
    }


class CsvLexer(RegexLexer):
    """Quoted CSV fields and the full-line comment convention of DCGM exporter."""

    tokens: ClassVar[dict] = {
        "root": [
            (r"^[ \t]*#[^\n]*", Comment),
            (r'"(?:[^"]|"")*"', String),
            (r",", Operator),
            (r"\bDCGM_FI_\w+", Name.Variable),
            (r"\b(?:gauge|counter|histogram|summary)\b", Keyword.Type),
            (r"[-+]?\b\d+(?:\.\d+)?\b", Number),
            (r'[^,#"\s]+|\s+|.', Text),
        ]
    }


SPECIAL = {"sass": SassLexer, "pseudocode": PseudocodeLexer, "csv": CsvLexer}
LABELS = {
    "sass": "SASS",
    "pseudocode": "算法伪代码",
    "csv": "CSV",
    "bash": "Shell",
    "cpp": "C++",
    "cuda": "CUDA C++",
    "ptx": "PTX",
    "diff": "Diff",
}
TOKEN_GROUPS = (
    (Generic.Inserted, "added"),
    (Generic.Deleted, "removed"),
    (Generic.Subheading, "type"),
    (Generic.Heading, "keyword"),
    (Comment.Preproc, "keyword"),
    (Comment.PreprocFile, "string"),
    (Comment, "comment"),
    (Keyword.Type, "type"),
    (Keyword, "keyword"),
    (Operator.Word, "keyword"),
    (Name.Class, "type"),
    (Name.Namespace, "type"),
    (Name.Tag, "variable"),
    (Name.Builtin, "type"),
    (Name.Function, "function"),
    (Name.Decorator, "function"),
    (Name.Label, "function"),
    (Name.Variable, "variable"),
    (Name.Constant, "number"),
    (String, "string"),
    (Number, "number"),
    (Operator, "operator"),
)


@lru_cache(None)
def lexer_for(language):
    if language in SPECIAL:
        return SPECIAL[language]()
    try:
        return get_lexer_by_name(language)
    except ClassNotFound as exc:
        raise ValueError(f"Unknown code language {language!r}") from exc


@lru_cache(None)
def default_style(token):
    return next((style for group, style in TOKEN_GROUPS if token in group), None)


def style_for(language, token, value, source, offset):
    if not value.strip():
        return None
    if language == "bash" and token is Text:
        if value.startswith("-"):
            return "keyword"
        if not source[source.rfind("\n", 0, offset) + 1 : offset].strip():
            return "function"
        if value.isdigit():
            return "number"
    if token in Name and re.match(
        r"[ \t]*\(", source[offset + len(value) : offset + len(value) + 32]
    ):
        return "function"
    if language == "ptx" and value.startswith(".") and token in Name:
        return "type"
    return default_style(token)


def highlight(figure):
    code = figure.find("pre/code")
    language = figure.get("data-language") or language_of(
        "".join(code.itertext()), figure.xpath("string(figcaption//a/@href)")
    )
    lexer = lexer_for(language)
    caption = figure.find("figcaption/span")
    label = caption.find("strong")
    if label is None:
        label = ET.Element("strong")
        label.tail, caption.text = caption.text, None
        caption.insert(0, label)
    label.text = LABELS.get(language, getattr(lexer, "name", language)) + "    "
    code.set("class", "language-" + language)
    rows = code.findall('span[@class="cl"]')
    original = ["".join(row.itertext()) for row in rows]
    source = "\n".join(original)
    styled = [[]]
    for offset, token, value in lexer.get_tokens_unprocessed(source):
        role = style_for(language, token, value, source, offset)
        for i, fragment in enumerate(value.split("\n")):
            if i:
                styled.append([])
            if fragment:
                line = styled[-1]
                if line and line[-1][0] == role:
                    line[-1] = (role, line[-1][1] + fragment)
                else:
                    line.append((role, fragment))
    if ["".join(s for _, s in line) for line in styled] != original:
        raise ValueError(f"{language} lexer changed source text or line breaks")
    for row, fragments in zip(rows, styled):
        for child in list(row):
            row.remove(child)
        row.text = None
        last = None
        for role, fragment in fragments:
            if role:
                last = ET.SubElement(row, "span", {"class": "tok-" + role})
                last.text = fragment
            elif last is None:
                row.text = fragment
            else:
                last.tail = fragment


def css():
    def rules(index):
        return "".join(
            f".code .tok-{role}{{color:#{pair[index]}}}"
            for role, pair in COLORS.items()
        )

    return rules(0) + "@media print{" + rules(1) + "}"
