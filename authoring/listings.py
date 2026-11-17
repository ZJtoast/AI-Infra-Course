"""Turn a source excerpt or a Markdown fence into a numbered course code block."""

import json
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from lxml import etree as ET
from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound

EXTENSIONS = {
    ".py": "python",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".ptx": "ptx",
    ".sass": "sass",
    ".csv": "csv",
    ".patch": "diff",
    ".diff": "diff",
    ".json": "json",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".ps1": "powershell",
}

CUDA_SYNTAX = re.compile(
    r"\b(?:__(?:global|shared|device|host)__|cuda[A-Z]\w*|cuda_runtime|cutlass|threadIdx|blockIdx)\b|<<<"
)
ALIASES = {
    "c++": "cpp",
    "cuda-cpp": "cuda",
    "shell": "bash",
    "sh": "bash",
    "py": "python",
    "pseudo": "pseudocode",
    "nvidia-sass": "sass",
}


def language_of(code, source="", hint=None):
    if hint:
        key = hint.lower()
        return ALIASES.get(key, key)
    path = PurePosixPath(unquote(urlsplit(source).path))
    ext = path.suffix.lower()
    if ext == ".h":
        gpu_headers = {"cuda", "__ptx", "flash_attn", "hopper", "blackwell", "device"}
        if CUDA_SYNTAX.search(code) or gpu_headers.intersection(path.parts):
            return "cuda"
        return "cpp"
    if ext in EXTENSIONS:
        return EXTENSIONS[ext]
    if ext:
        try:
            return get_lexer_for_filename(path.name, code).aliases[0]
        except ClassNotFound:
            pass
    if re.search(r"(?m)^\s*\.(?:version|target|entry|address_size)\b", code):
        return "ptx"
    if re.search(r"(?m)^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?\w+\s+)?[A-Z][A-Z0-9.]+\b", code):
        return "sass"
    if re.search(
        r"(?im)^\s*(?:WITH\s+\w+\s+AS\s*\(|SELECT\s+.+\s+FROM\b|CREATE\s+TABLE\b)", code
    ):
        return "sql"
    if re.search(
        r"(?m)^\s*(?:#!.*\b(?:ba)?sh\b|(?:ncu|nsys|nvcc|ptxas|nvdisasm|git|pip|cmake|make|export|echo|python3?)\s+|\./[\w/-]+)",
        code,
    ):
        return "bash"
    if CUDA_SYNTAX.search(code):
        return "cuda"
    if re.search(
        r"(?m)^\s*(?:from\s+\S+\s+import |import\s+[\w.]+(?:\s+as\s+\w+)?\s*$|(?:async\s+)?def\s+\w+\(|with\s+.+:\s*$)|\b(?:torch|numpy|np|cp)\.\w+\(",
        code,
    ):
        return "python"
    if re.search(
        r"←|\b(?:for each|parallel for)\b|(?m:^\s*(?:(?:producer|consumer)|\w+ warp):)",
        code,
    ):
        return "pseudocode"
    # Instructional algorithms may consist entirely of instructions in prose.
    operations = re.findall(
        r"(?m)^\s*(?:choose|bound|quantize|issue|reset|restore|compute|load|store)\s+(?:the|a|each|block|Q|K|V)\b",
        code,
    )
    if len(operations) >= 3:
        return "pseudocode"
    if code.lstrip().startswith(("{", "[")):
        try:
            json.loads(code)
            return "json"
        except ValueError:
            pass
    return guess_lexer(code).aliases[0]


def unfence(value):
    # Leave indentation and interior blank lines untouched, including edge blanks inside a fence.
    match = re.fullmatch(
        r"\s*?(?P<fence>\x60{3,}|~{3,})[ \t]*(?P<lang>[\w.+#-]*)[ \t]*\n(?P<code>.*?)\n(?P=fence)[ \t]*\s*",
        value,
        re.DOTALL,
    )
    if match:
        return match["code"], match["lang"]
    return value.removeprefix("\n").removesuffix("\n"), None


def expand(listing):
    source = listing.get("src", "")
    code, hint = unfence(listing.text or "")
    lang = language_of(code, source, hint)
    fragment = urlsplit(source).fragment
    line_ref = re.match(r"L(\d+)", fragment)
    first = int(listing.get("start", line_ref[1] if line_ref else "1"))
    figure = ET.Element("figure", {"class": "code", "data-language": lang})
    if listing.get("id"):
        figure.set("id", listing.get("id"))
    header = ET.SubElement(figure, "figcaption")
    caption = ET.SubElement(header, "span")
    if source:
        ET.SubElement(caption, "a", href=source).text = (
            listing.get("title") or PurePosixPath(urlsplit(source).path).name
        )
    else:
        caption.text = listing.get("title")
    ET.SubElement(
        header, "button", {"type": "button", "data-copy": ""}
    ).text = "复制代码"
    block = ET.SubElement(ET.SubElement(figure, "pre"), "code")
    lines = code.split("\n")
    notes = listing.findall("annotations/line")
    if lang == "pseudocode":
        expected = [i for i, line in enumerate(lines, first) if line.strip()]
        if [int(note.get("n")) for note in notes] != expected or any(
            not re.search(r"[\u3400-\u9fff]", note.text or "") for note in notes
        ):
            raise ValueError(f"Every pseudocode instruction needs a numbered Chinese explanation: {listing.get('title')}")
    by_number = {int(note.get("n")): note.text for note in notes}
    for i, line in enumerate(lines, first):
        row = ET.SubElement(block, "span", {"class": "cl", "data-line": str(i)})
        row.text = line
        if i in by_number:
            row.set("data-explanation", by_number[i])
        row.tail = "\n"
    block[-1].tail = None
    figure.tail = listing.tail
    listing.getparent().replace(listing, figure)
