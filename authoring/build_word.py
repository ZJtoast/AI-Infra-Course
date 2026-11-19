"""Render editable Word equations, bookmarks, source excerpts and print figures."""

import os
import re
from functools import lru_cache
from equations import aligned_dimensions
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from course import (
    ET,
    HERE,
    args,
    image_bytes,
    image_path,
    read,
    text,
    topic_groups,
    write,
)
from PIL import Image
from source_highlight import COLORS

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
PIC = "{http://schemas.openxmlformats.org/drawingml/2006/picture}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def w(parent, tag, **attrs):
    return ET.SubElement(parent, W + tag, {W + k: str(v) for k, v in attrs.items()})


def xml(e):
    return ET.tostring(e, encoding="utf-8", xml_declaration=True, standalone=True)


def anchor(value):
    match = re.fullmatch(r"(idx-)?l(\d+)-s(\d+)", value)
    if match:
        return ("Index" if match[1] else "Topic") + match[2] + "_" + match[3]
    return (
        value
        if re.fullmatch(r"[A-Za-z]\w{0,39}", value)
        else "Ref" + sha1(value.encode()).hexdigest()[:32]
    )


@lru_cache(None)
def stylesheet():
    candidates = [Path(os.environ.get("MML2OMML_XSL", "MML2OMML.XSL"))]
    candidates += [
        Path(os.environ.get(k, "C:/Program Files"))
        / "Microsoft Office/root/Office16/MML2OMML.XSL"
        for k in ("ProgramFiles", "ProgramFiles(x86)")
    ]
    for f in candidates:
        if f.is_file():
            return ET.XSLT(ET.parse(f))
    raise FileNotFoundError(
        "Set MML2OMML_XSL to the MathML to OMML stylesheet supplied with Microsoft Word."
    )


@lru_cache(maxsize=5000)
def formula(source):
    math = ET.fromstring(source)
    aligned_dimensions(math)
    equation = stylesheet()(math).getroot()
    # Office's transform treats a MathML underbrace as a fixed-width limit.
    # Use a stretching brace while retaining the separately editable dimension.
    for limit in equation.iterfind(f".//{M}limLow"):
        if limit.findtext(f"{M}lim/{M}r/{M}t") == "⏟":
            group = ET.Element(M + "groupChr")
            properties = ET.SubElement(group, M + "groupChrPr")
            for name, value in (("chr", "⏟"), ("pos", "bot"), ("vertJc", "top")):
                ET.SubElement(properties, M + name, {M + "val": value})
            group.append(limit.find(M + "e"))
            limit.getparent().replace(limit, group)
    return ET.tostring(equation)


def paragraph(parent, style=None, indent=False, keep=False):
    p = w(parent, "p")
    pr = w(p, "pPr")
    if style:
        w(pr, "pStyle", val=style)
    if keep:
        w(pr, "keepNext")
    if indent:
        w(pr, "ind", firstLine="420")
    return p


def run(
    parent,
    value,
    bold=False,
    italic=False,
    script=None,
    color=None,
    mono=False,
    dotted=False,
):
    if not value:
        return
    r = w(parent, "r")
    pr = w(r, "rPr")
    font = "Consolas" if mono else "Times New Roman"
    w(pr, "rFonts", ascii=font, hAnsi=font)
    if bold:
        w(pr, "b")
    if italic:
        w(pr, "i")
    if color:
        w(pr, "color", val=color)
    if dotted:
        w(pr, "u", val="dotted")
    if script:
        w(pr, "vertAlign", val=script)
    t = w(r, "t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = value


class Document:
    def __init__(self, chapters, book=False):
        self.chapters = chapters
        self.book = book
        self.sequence = 0
        self.keys = set()
        self.targets = set()
        self.defs = {
            dt.get("id"): text(dt.getnext()) for c in chapters for dt in c.iter("dt")
        }
        road = chapters[0].get("slug").startswith("00-")
        with ZipFile(
            HERE / ("roadmap-layout.docx" if road else "lesson-layout.docx")
        ) as z:
            self.parts = {n: z.read(n) for n in z.namelist()}
        styles = ET.fromstring(self.parts["word/styles.xml"])
        equation_size = styles.find(
            f"{W}style[@{W}styleId='EquationBlock']/{W}rPr/{W}sz"
        )
        if equation_size is not None:
            equation_size.set(W + "val", "30")  # Word uses half-points.
            self.parts["word/styles.xml"] = xml(styles)
        self.root = ET.fromstring(self.parts["word/document.xml"])
        self.body = self.root.find(W + "body")
        self.sect = self.body[-1]
        self.body.remove(self.sect)
        self.rels = ET.fromstring(self.parts["word/_rels/document.xml.rels"])
        self.relmap = {}
        self.types = ET.fromstring(self.parts["[Content_Types].xml"])
        if not any(x.get("Extension") == "png" for x in self.types):
            ET.SubElement(
                self.types,
                self.types.tag.replace("Types", "Default"),
                Extension="png",
                ContentType="image/png",
            )

    def rel(self, kind, target):
        key = (kind, target)
        if key not in self.relmap:
            rid = "rIdCourse" + str(len(self.relmap) + 1)
            attrs = {"Id": rid, "Type": R[1:-1] + "/" + kind, "Target": target}
            if kind == "hyperlink":
                attrs["TargetMode"] = "External"
            ET.SubElement(self.rels, PKG + "Relationship", attrs)
            self.relmap[key] = rid
        return self.relmap[key]

    def mark(self, p, key):
        if not key:
            return
        k = anchor(key)
        if k in self.keys:
            raise ValueError("Duplicate Word bookmark " + k)
        self.keys.add(k)
        self.sequence += 1
        start = w(p, "bookmarkStart", id=self.sequence, name=k)
        p.insert(1, start)
        w(p, "bookmarkEnd", id=self.sequence)

    def link(self, p, href, label=None):
        a = w(p, "hyperlink")
        if href.startswith("#"):
            target = href[1:]
            self.targets.add(anchor(target))
            a.set(W + "anchor", anchor(target))
            if target in self.defs:
                a.set(W + "tooltip", self.defs[target][:250])
        else:
            if href.startswith("examples/"):
                href = "https://github.com/ZJtoast/AI-Infra-Course/blob/main/" + href
            elif href.startswith(("lessons/", "handbook.html#")):
                href = "https://zjtoast.github.io/AI-Infra-Course/" + href
            a.set(R + "id", self.rel("hyperlink", href))
        if label:
            run(a, label, color="14645A")
        return a

    def inline(self, parent, node, **opts):
        run(parent, node.text, **opts)
        for e in node:
            tag = ET.QName(e).localname
            fmt = dict(opts)
            dest = parent
            if tag == "math":
                parent.append(ET.fromstring(formula(ET.tostring(e, with_tail=False))))
            elif tag == "br":
                w(w(parent, "r"), "br")
            elif tag != "button":
                if tag == "a":
                    dest = self.link(parent, e.get("href"))
                    fmt["color"] = "14645A"
                if tag == "a" and e.get("class") == "term":
                    fmt["dotted"] = True
                    fmt["bold"] = True
                if tag == "a" and e.get("class") in ("example-ref", "line-ref"):
                    fmt["color"] = "185CA4" if e.get("class") == "example-ref" else "A13232"
                if tag == "span" and opts.get("mono"):
                    role = e.get("class", "").removeprefix("tok-")
                    if role in COLORS:
                        fmt["color"] = COLORS[role][0]
                if tag in ("strong", "b"):
                    fmt["bold"] = True
                if tag in ("em", "i"):
                    fmt["italic"] = True
                if tag == "code":
                    fmt["mono"] = True
                if tag in ("sup", "sub"):
                    fmt["script"] = "superscript" if tag == "sup" else "subscript"
                self.inline(dest, e, **fmt)
            run(parent, e.tail, **opts)

    def image(self, im, parent):
        f = image_path(im.get("src"))
        target = "media/" + f.name
        if "word/" + target not in self.parts:
            self.parts["word/" + target] = image_bytes(f)
        with Image.open(BytesIO(self.parts["word/" + target])) as bitmap:
            width, height = bitmap.size
        cx = min(5610000, int(7620000 * width / height))
        cy = round(cx * height / width)
        p = paragraph(parent, keep=True)
        w(p[0], "jc", val="center")
        drawing = w(w(p, "r"), "drawing")
        frame = ET.SubElement(drawing, WP + "inline")
        ET.SubElement(frame, WP + "extent", cx=str(cx), cy=str(cy))
        self.sequence += 1
        ET.SubElement(
            frame,
            WP + "docPr",
            id=str(self.sequence),
            name=f.name,
            descr=im.get("alt", ""),
        )
        data = ET.SubElement(
            ET.SubElement(frame, A + "graphic"), A + "graphicData", uri=PIC[1:-1]
        )
        picture = ET.SubElement(data, PIC + "pic")
        nv = ET.SubElement(picture, PIC + "nvPicPr")
        ET.SubElement(nv, PIC + "cNvPr", id="0", name=f.name)
        ET.SubElement(nv, PIC + "cNvPicPr")
        fill = ET.SubElement(picture, PIC + "blipFill")
        ET.SubElement(fill, A + "blip", {R + "embed": self.rel("image", target)})
        ET.SubElement(ET.SubElement(fill, A + "stretch"), A + "fillRect")
        shape = ET.SubElement(picture, PIC + "spPr")
        xfrm = ET.SubElement(shape, A + "xfrm")
        ET.SubElement(xfrm, A + "off", x="0", y="0")
        ET.SubElement(xfrm, A + "ext", cx=str(cx), cy=str(cy))
        ET.SubElement(ET.SubElement(shape, A + "prstGeom", prst="rect"), A + "avLst")

    def table(self, parent, rows, code=False):
        tbl = w(parent, "tbl")
        pr = w(tbl, "tblPr")
        w(pr, "tblStyle", val="TableGrid")
        w(pr, "tblW", w="5000", type="pct")
        if code:
            number_width = max(len(row.get("data-line", "")) for row in rows)
            borders = w(pr, "tblBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                w(borders, edge, val="nil")
        grid = w(tbl, "tblGrid")
        cols = 1 if code else max(len(r) for r in rows)
        for _ in range(cols):
            w(grid, "gridCol", w=9250 // cols)
        for i, row in enumerate(rows):
            tr = w(tbl, "tr")
            trpr = w(tr, "trPr")
            w(trpr, "cantSplit")
            if i == 0 and not code:
                w(trpr, "tblHeader")
            for c in [row] if code else row:
                cell = w(tr, "tc")
                cp = w(cell, "tcPr")
                w(cp, "tcW", w=9250 // cols, type="dxa")
                if code or i == 0:
                    w(cp, "shd", fill="18303A" if code else "E2EEE9")
                p = paragraph(cell, "SourceCode" if code else None)
                if code:
                    w(p[0], "spacing", before=0, after=0)
                    run(
                        p,
                        c.get("data-line", "").rjust(number_width) + "    ",
                        mono=True,
                        color="94ABA8",
                    )
                    self.inline(p, c, mono=True, color="ECF3F2")
                else:
                    self.inline(p, c, bold=c.tag == "th")

    def block(self, parent, e):
        tag = e.tag
        cls = e.get("class", "")
        if tag == "p":
            p = paragraph(
                parent,
                "Caption"
                if cls == "caption"
                else "EquationBlock"
                if cls == "eq"
                else None,
                indent=cls not in ("eq", "caption"),
            )
            self.mark(p, e.get("id"))
            self.inline(p, e)
        elif tag in ("h2", "h3"):
            p = paragraph(parent, "Heading1" if tag == "h2" else "Heading2", keep=True)
            self.inline(p, e)
            self.mark(p, e.getparent().get("id"))
        elif tag == "exercise":
            p = paragraph(parent, keep=True)
            self.mark(p, e.get("id"))
            run(p, e.get("kind") + "　" + e.get("title") + "　", bold=True)
            self.link(p, "#" + e.find("answer").get("id"), "查看推导")
            for child in e:
                self.block(parent, child)
        elif tag == "answer":
            p = paragraph(parent, keep=True)
            self.mark(p, e.get("id"))
            run(p, "参考解答　", bold=True)
            self.link(p, "#" + e.getparent().get("id"), "回看题干")
            for child in e:
                self.block(parent, child)
        elif tag == "nav":
            p = paragraph(parent, keep=True)
            self.mark(p, e.get("id"))
            run(p, "本课练习顺序", bold=True)
            for li in e.iter("li"):
                self.inline(paragraph(parent), li)
        elif tag == "table":
            self.table(parent, e.xpath("./tr|./thead/tr|./tbody/tr"))
        elif tag == "figure" and cls == "code":
            caption = paragraph(parent, "SourceCaption", keep=True)
            self.mark(caption, e.get("id"))
            self.inline(caption, e.find("figcaption"))
            self.table(parent, e.xpath('.//span[@class="cl"]'), True)
        elif tag == "figure":
            for im in e.iter("img"):
                self.image(im, parent)
            for cap in e.iter("figcaption"):
                self.inline(paragraph(parent, "Caption"), cap)
        elif "state-deck" in cls.split():
            run(paragraph(parent, keep=True), text(e.find("h4")), bold=True)
            for frame in e.findall("details"):
                p = paragraph(parent, keep=True)
                self.mark(p, frame.get("id"))
                run(p, text(frame.find("summary")), bold=True)
                self.image(frame.find(".//img"), parent)
                self.inline(paragraph(parent, "Caption"), frame.find("p"))
                for t in frame.iter("table"):
                    self.block(parent, t)
        elif cls == "matrix-lab":
            p = paragraph(parent, keep=True)
            self.mark(p, e.get("id"))
            run(p, text(e.find("h4")), bold=True)
            for note in e.findall("p"):
                self.block(parent, note)
            im = e.find('img[@class="mx-print"]')
            self.image(im, parent)
            run(paragraph(parent, "Caption"), im.get("alt"))
        elif e.get("data-sass-lab") is not None:
            self.link(paragraph(parent),
                      "https://zjtoast.github.io/AI-Infra-Course/handbook.html#l18-sm80-source-lab",
                      "打开 SM80 指令与源码对照")
        elif cls == "counter-reader":
            self.link(paragraph(parent),
                      "https://zjtoast.github.io/AI-Infra-Course/handbook.html#l18-counter-catalog",
                      "打开 GA100 硬件指标查询表")
        elif cls == "definitions":
            run(paragraph(parent, "Heading2"), text(e.find("summary")), bold=True)
            for dt in e.iter("dt"):
                p = paragraph(parent)
                self.mark(p, dt.get("id"))
                run(p, text(dt) + "　", bold=True)
                self.inline(p, dt.getnext(), bold=True)
        elif tag == "section" or cls == "tablewrap":
            for child in e:
                self.block(parent, child)
            if cls == "topic":
                p = paragraph(parent)
                w(p[0], "jc", val="right")
                self.link(p, "#idx-" + e.get("id"), "↑ 返回知识点目录")
        elif tag != "lead":
            raise ValueError("Unsupported course block: " + str(tag) + " " + cls)

    def build(self):
        title = (
            "AI Infra Course · 知识点导航"
            if self.book
            else self.chapters[0].get("title")
        )
        metadata = ET.fromstring(self.parts["docProps/core.xml"])
        dc = "{http://purl.org/dc/elements/1.1/}"
        metadata.find(dc + "title").text = title
        metadata.find(dc + "subject").text = (
            "AI Infra Course" if self.book else self.chapters[0].get("group")
        )
        metadata.find(dc + "description").text = (
            "AI Infra Course 课程合订本"
            if self.book
            else text(self.chapters[0].find("lead"))
        )
        self.parts["docProps/core.xml"] = xml(metadata)
        run(paragraph(self.body, "Title"), title)
        for ch in self.chapters:
            run(
                paragraph(self.body, "Heading1", keep=True),
                ch.get("title") if self.book else "知识点目录",
            )
            for group, sections in topic_groups(ch):
                if group:
                    run(paragraph(self.body, keep=True), group, bold=True)
                for s in sections:
                    p = paragraph(self.body)
                    self.mark(p, "idx-" + s.get("id"))
                    self.link(p, "#" + s.get("id"), text(s[0]))
        for ch in self.chapters:
            p = paragraph(self.body, "Heading1", keep=True)
            w(p[0], "pageBreakBefore")
            run(p, ch.get("title"))
            self.inline(paragraph(self.body, indent=True), ch.find("lead"))
            for e in ch:
                self.block(self.body, e)
        missing = self.targets - self.keys
        if missing:
            raise ValueError("Unresolved Word bookmarks: " + str(sorted(missing)[:10]))
        self.body.append(self.sect)
        self.parts["word/document.xml"] = xml(self.root)
        self.parts["word/_rels/document.xml.rels"] = xml(self.rels)
        self.parts["[Content_Types].xml"] = xml(self.types)
        buff = BytesIO()
        with ZipFile(buff, "w") as z:
            for name, data in self.parts.items():
                z.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
        return buff.getvalue()


def build(chapters, out):
    for ch in chapters:
        write(out / "word" / (ch.get("slug") + ".docx"), Document([ch]).build())
    write(
        out / "word/ai-infra-course-handbook.docx",
        Document(chapters[1:], book=True).build(),
    )
    print(f"{len(chapters) + 1} Word documents")


if __name__ == "__main__":
    opt = args()
    build(read(opt.source), opt.output)
