"""Render the reader, offline handbook, course homepage and search index."""

import json
import re
import shutil
from copy import deepcopy
from html import escape
from io import BytesIO

from course import (
    ET,
    GENERATED,
    HERE,
    ROOT,
    args,
    chapter_url,
    digest,
    image_bytes,
    image_path,
    read,
    text,
    topics,
    write,
)
from lxml import html
from listings import language_of
from PIL import Image
from source_highlight import css as syntax_css, language_label, styled_lines


def markup(node):
    return ET.tostring(node, encoding="unicode", method="html", with_tail=False)


def link(url, label, cls=""):
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(label)}</a>'


def prepare_line_refs(chapter):
    # Keep the lexer's context for slices that start inside a string or a comment.
    excerpts = {}
    languages = {}
    source_rows = {}
    for fig in chapter.xpath('.//figure[@class="code"]'):
        url = fig.xpath("string(figcaption//a/@href)").partition("#")[0]
        keys = [k for k in (url, "#" + fig.get("id") if fig.get("id") else "") if k]
        for key in keys:
            languages[key] = fig.get("data-language")
        for row in fig.xpath('.//span[@class="cl"]'):
            parts = [(None, row.text)] if row.text else []
            for span in row:
                if fig.get("data-language") == "pseudocode" and span.getnext() is None and (span.text or "").startswith("  // "):
                    break
                parts.append((span.get("class").removeprefix("tok-"), span.text or ""))
                if span.tail:
                    parts.append((None, span.tail))
            original = "".join(value for _, value in parts)
            for key in keys:
                position = (key, int(row.get("data-line")), original)
                excerpts[position] = parts
                if fig.get("data-language") != "pseudocode":
                    source_rows.setdefault(position, []).append(row)
    for link in chapter.xpath('.//a[@class="line-ref"]'):
        href = link.get("href")
        key = href if href.startswith("#") else href.partition("#")[0]
        rows = json.loads(link.get("data-rows"))
        lines = [source for _, source, _ in rows]
        language = languages.get(key) or language_of("\n".join(lines), href)
        parts = [excerpts.get((key, number, source)) for number, source, _ in rows]
        if any(line is None for line in parts):
            fallback = styled_lines(language, lines)
            parts = [original if original is not None else replacement for original, replacement in zip(parts, fallback)]
        link.set("data-language", language)
        link.set("data-language-label", language_label(language))
        link.set("data-syntax", json.dumps(parts, ensure_ascii=False, separators=(",", ":")))
        for number, source, explanation in rows:
            for row in source_rows.get((key, number, source), []):
                if source.strip() and "data-reading-note" not in row.attrib:
                    row.set("data-reading-note", link.get("data-comment") + " " + explanation)

    for fig in chapter.xpath('.//figure[@class="code"][.//*[@data-reading-note]]'):
        caption = fig.find("figcaption")
        controls = ET.SubElement(caption, "span", {"class": "code-controls"})
        toggle = ET.SubElement(controls, "button", {
            "type": "button", "data-toggle-notes": "", "aria-pressed": "false"
        })
        toggle.text = f"行注（{len(fig.xpath('.//*[@data-reading-note]'))} 行）"
        controls.append(caption.find("button[@data-copy]"))


def practice(q, names, route):
    key = q.get("id")
    ans = q.find("answer")
    body = "".join(markup(c) for c in q if c is not ans)
    target = q.get("next")
    nextlink = link("#" + target, "下一题 · " + names[target]) if target else ""
    return (
        f'<div class="exercise question" id="{key}" role="group" aria-labelledby="title-{key}">'
        f'<div class="qtitle" id="title-{key}"><span class="exercise-no">{escape(q.get("kind"))}</span>'
        f"<strong>{escape(q.get('title'))}</strong></div>"
        + body
        + f'<div class="response"><div class="response-head"><label for="draft-{key}">解题笔记</label>'
        '<span class="draft-status" role="status"></span></div>'
        f'<textarea id="draft-{key}" data-exercise="{key}" rows="3" aria-describedby="prompt-{key}"></textarea></div>'
        f'<details class="solution" id="{ans.get("id")}"><summary>答案</summary><div class="solution-body">'
        + "".join(map(markup, ans))
        + '<div class="practice-links">'
        + link("#" + key, "回看题干 ↑", "return-question")
        + nextlink
        + "　　"
        + link("#" + route, "本课练习顺序")
        + "</div></div></details></div>"
    )


def contents(ch, prefix):
    copy = deepcopy(ch)
    # MathML Core uses explicit operators for fences, including ceiling and floor.
    for fenced in copy.xpath('.//*[local-name()="mfenced"]'):
        ns = "{http://www.w3.org/1998/Math/MathML}"
        children = list(fenced)
        opening = fenced.attrib.pop("open", "(")
        closing = fenced.attrib.pop("close", ")")
        separators = "".join(fenced.attrib.pop("separators", ",").split())
        fenced.tag = ns + "mrow"
        left = ET.Element(ns + "mo", fence="true")
        left.text = opening
        fenced.insert(0, left)
        if separators:
            for i, child in enumerate(children[1:]):
                separator = ET.Element(ns + "mo", separator="true")
                separator.text = separators[min(i, len(separators) - 1)]
                child.addprevious(separator)
        ET.SubElement(fenced, ns + "mo", fence="true").text = closing
    names = {q.get("id"): q.get("title") for q in ch.iter("exercise")}
    for e in copy.iter():
        for k in ("href", "src"):
            u = e.get(k, "")
            if u.startswith("assets/figures/"):
                f = image_path(u)
                e.set(k, prefix + u + "?v=" + digest(f))
                if e.tag == "img":
                    with Image.open(BytesIO(image_bytes(f))) as im:
                        e.set("width", str(im.width))
                        e.set("height", str(im.height))
            elif u.startswith("examples/"):
                e.set(k, prefix + u)
    for q in copy.xpath(".//exercise"):
        # Replacements stay in the original section next to the material they test.
        replacement = html.fragment_fromstring(
            practice(q, names, "Route" + ch.get("slug")[:2])
        )
        q.getparent().replace(q, replacement)
    for sec in topics(copy):
        back = ET.SubElement(
            sec, "a", {"class": "back", "href": "#idx-" + sec.get("id")}
        )
        back.text = "↑ 返回知识点目录"
    return "".join(markup(e) for e in copy if e.tag != "lead")


def toc(ch):
    return (
        '<ol class="toc">'
        + "".join(
            f'<li class="lv{s[0].tag[1]}" id="idx-{s.get("id")}">'
            + link("#" + s.get("id"), text(s[0]))
            + "</li>"
            for s in topics(ch)
        )
        + "</ol>"
    )


def reader(title, eyebrow, lead, download, aside, body, prefix):
    s = (
        (HERE / "reader.html")
        .read_text("utf8")
        .replace("</head>", f"<style>{syntax_css()}</style></head>")
    )
    s = re.sub(
        r'((?:href|src)=")(?=(?:assets/|index.html|handbook.html|roadmap.html))',
        r"\g<1>" + prefix,
        s,
    )
    s = s.replace('data-root=""', 'data-root="' + prefix + '"')
    hero = (
        f'<div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1><p>{escape(lead)}</p><div class="actions">'
        + link(prefix + "word/" + download + ".docx", "下载 Word ↓", "btn")
        + "</div>"
    )
    return (
        s.replace("{{TITLE}}", escape(title) + " · AI Infra Course")
        .replace("{{HERO}}", hero)
        .replace("{{ASIDE}}", aside)
        .replace("{{BODY}}", body)
    )


def build(chapters, out):
    files = {}
    book = []
    index = []
    chapnav = []
    groups = {}
    for i, ch in enumerate(chapters):
        prepare_line_refs(ch)
        slug = ch.get("slug")
        title = ch.get("title")
        lead = text(ch.find("lead"))
        group = ch.get("group")
        prefix = "../" if i else ""
        nav = toc(ch)
        adjacent = (
            '<div class="adjacent">'
            + "".join(
                link(
                    prefix + chapter_url(chapters[j]), arrow + chapters[j].get("title")
                )
                for j, arrow in [(i - 1, "← "), (i + 1, "→ ")]
                if 0 <= j < len(chapters)
            )
            + "</div>"
        )
        files[chapter_url(ch)] = reader(
            title,
            f"第 {slug[:2]} 课　/　{group}" if i else group,
            lead,
            slug,
            f'<details class="tocbox" open><summary>本页知识点 <small>{len(topics(ch))}</small></summary>{nav}</details>',
            contents(ch, prefix) + adjacent,
            prefix,
        )
        index.append(
            {
                "n": i,
                "title": title,
                "url": chapter_url(ch),
                "items": [
                    [
                        s.get("id"),
                        text(s[0]),
                        " ".join(text(p) for p in s.findall("p"))[:260],
                    ]
                    for s in topics(ch)
                ],
            }
        )
        if i:
            groups.setdefault(group, []).append(ch)
            book.append(
                f'<header class="chapterhead" id="ch-{slug[:2]}"><div class="eyebrow">第 {slug[:2]} 课</div><h2>{escape(title)}</h2></header>'
                + contents(ch, "")
            )
            chapnav.append(
                f'<details class="chap"><summary>{slug[:2]}　{escape(title)}</summary>{nav}</details>'
            )
    files["handbook.html"] = reader(
        "知识点导航",
        "全课程合订本",
        "展开对应课程，选择知识点即可跳到正文。每节末尾可以返回该条目录项。",
        "ai-infra-course-handbook",
        '<div class="booknav">' + "".join(chapnav) + "</div>",
        "".join(book),
        "",
    )
    home = (HERE / "home.html").read_text("utf8")

    def row(ch):
        return f'<a class="lessonrow" href="{chapter_url(ch)}"><span class="num">{ch.get("slug")[:2]}</span><div><h3>{escape(ch.get("title"))}</h3><p>{escape(text(ch.find("lead")))}</p></div><span class="go" aria-hidden="true">↗</span></a>'

    counts = [
        (len(chapters) - 1, "课"),
        (sum(len(topics(c)) for c in chapters[1:]), "个知识点"),
        (sum(len(list(c.iter("exercise"))) for c in chapters), "道随堂练习"),
    ]
    files["index.html"] = home.replace(
        "{{METRICS}}",
        "".join(f"<span><b>{n}</b> {label}</span>" for n, label in counts),
    ).replace(
        "<!--GROUPS-->",
        "".join(
            f'<section class="group" id="module-{i}"><div class="grouphead"><small>模块 {i}</small><h2>{escape(g)}</h2></div><div class="grouprows">'
            + "".join(map(row, items))
            + "</div></section>"
            for i, (g, items) in enumerate(groups.items(), 1)
        ),
    )
    js = (
        "window.courseIndex="
        + json.dumps(index, ensure_ascii=False, separators=(",", ":"))
        + ";"
    )
    write(out / "assets/index-data.js", js)
    if out.resolve() != ROOT:
        shutil.copytree(
            ROOT / "assets",
            out / "assets",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("index-data.js"),
        )
        shutil.copytree(ROOT / "examples", out / "examples", dirs_exist_ok=True)
    for name, data in GENERATED.items():
        write(out / "assets/figures" / name, data)
    for name, page in files.items():
        for asset in ("course.css", "course.js", "index-data.js"):
            page = re.sub(
                re.escape("assets/" + asset) + r"\?v=\w+",
                "assets/" + asset + "?v=" + digest(out / "assets" / asset),
                page,
            )
        write(out / name, page)
    print(f"{len(files)} HTML pages and search index")


if __name__ == "__main__":
    opt = args()
    build(read(opt.source), opt.output)
