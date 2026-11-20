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
    topic_groups,
    topics,
    write,
)
from lxml import html
from equations import MATH, aligned_dimensions
from symbols import prepare_symbol_notes
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
    figures_by_row = {}
    for index, fig in enumerate(chapter.xpath('.//figure[@class="code"]'), 1):
        if not fig.get("id"):
            fig.set("id", f"code-{chapter.get('slug')[:2]}-{index}")
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
                figures_by_row.setdefault(position, set()).add(fig.get("id"))
                excerpts[position] = parts
                if fig.get("data-language") != "pseudocode":
                    source_rows.setdefault(position, []).append(row)
    for link in chapter.xpath('.//a[@class="line-ref"]'):
        href = link.get("href")
        key = href if href.startswith("#") else href.partition("#")[0]
        rows = json.loads(link.get("data-rows"))
        lines = [source for _, source, _ in rows]
        language = languages.get(key) or language_of("\n".join(lines), href)
        positions = [(key, number, source) for number, source, _ in rows]
        figures = set.intersection(*(figures_by_row.get(pos, set()) for pos in positions))
        if figures:
            link.set("data-source-figures", " ".join(sorted(figures)))
            link.set("data-first-line", str(rows[0][0]))
            link.set("data-last-line", str(rows[-1][0]))
        parts = [excerpts.get(pos) for pos in positions]
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
    for math in copy.iter(MATH + "math"):
        aligned_dimensions(math)
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
            practice(q, names, ch.find("nav[@class='practice-route']").get("id"))
        )
        q.getparent().replace(q, replacement)
    for sec in topics(copy):
        back = ET.SubElement(
            sec, "a", {"class": "back", "href": "#idx-" + sec.get("id")}
        )
        back.text = "↑ 返回知识点目录"
    return "".join(markup(e) for e in copy if e.tag != "lead")


def reading_stops(topic):
    return (topic, *topic.findall('div[@class="subtopic"]'))


def toc(ch, page=""):
    groups = []
    for name, sections in topic_groups(ch):
        entries = "".join(
            f'<li class="lv{s[0].tag[1]}" id="idx-{s.get("id")}">'
            + link(page + "#" + s.get("id"), text(s[0]))
            + "</li>"
            for topic in sections for s in reading_stops(topic)
        )
        if name:
            groups.append(f"<strong>{escape(name)}</strong>")
        groups.append(f'<ol class="toc">{entries}</ol>')
    return "".join(groups)


def course_navigation(groups, current=None, *, handbook=False):
    modules = []
    for name, chapters in groups.items():
        lessons = []
        for ch in chapters:
            number = ch.get("slug")[:2]
            local = ch is current
            page = "" if handbook or local else chapter_url(ch).removeprefix("lessons/")
            anchor = "#ch-" + number if handbook else (page or "#main")
            lessons.append(
                f'<details class="chap" data-chapter="{number}"{" open" if local and not handbook else ""}>'
                f'<summary><span class="chapter-number">{ch.get("number")}</span><span>{escape(ch.get("title"))}</span></summary>'
                + link(anchor, "阅读本课") + toc(ch, page) + "</details>"
            )
        opened = " open" if any(ch is current for ch in chapters) else ""
        modules.append(
            f'<details class="toc-module"{opened}><summary>{escape(name)}</summary>'
            + "".join(lessons) + "</details>"
        )
    return '<nav class="booknav" aria-label="按主题浏览课程">' + "".join(modules) + "</nav>"


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
    labs = {
        "data-ncu-catalog": ("ncu-ga100.js", "ncu-counter-reader.js"),
        "data-sass-lab": ("ncu-sm80-code.js", "ncu-sass-reader.js"),
    }
    scripts = "".join(
        f'<script defer src="{prefix}assets/{file}?v={digest(ROOT / "assets" / file)}"></script>'
        for marker, files in labs.items() if marker in body for file in files
    )
    s = s.replace("</head>", scripts + "</head>")
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
    groups = {}
    for position, ch in enumerate(chapters[1:], 1):
        ch.set("number", f"{position:02d}")
        groups.setdefault(ch.get("group"), []).append(ch)
    for i, ch in enumerate(chapters):
        prepare_line_refs(ch)
        prepare_symbol_notes(ch)
        slug = ch.get("slug")
        title = ch.get("title")
        lead = text(ch.find("lead"))
        group = ch.get("group")
        prefix = "../" if i else ""
        adjacent = (
            '<div class="adjacent">'
            + "".join(
                link(
                    prefix + chapter_url(chapters[j]), arrow + chapters[j].get("title")
                )
                for j, arrow in [(i - 1, "← " if i == 1 else "← 上一课　"), (i + 1, "下一课　")]
                if 0 <= j < len(chapters)
            )
            + "</div>"
        )
        files[chapter_url(ch)] = reader(
            title,
            f"第 {i:02d} 课　/　{group}" if i else group,
            lead,
            slug,
            course_navigation(groups, ch) if i else
            f'<details class="tocbox" open><summary>路线站点</summary>{toc(ch)}</details>',
            contents(ch, prefix) + adjacent,
            prefix,
        )
        legacy = ch.get("legacy-slug")
        if legacy:
            dest = slug + ".html"
            files["lessons/" + legacy + ".html"] = (
                '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                f'<title>{escape(title)}</title>'
                f'<script>location.replace("{dest}" + location.search + location.hash)</script>'
                f'<a href="{dest}">{escape(title)}</a></html>'
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
                    for topic in topics(ch) for s in reading_stops(topic)
                ],
            }
        )
        if i:
            neighbors = "".join(
                link("#ch-" + chapters[j].get("slug")[:2], label + chapters[j].get("title"))
                for j, label in ((i - 1, "← 上一课　"), (i + 1, "下一课　"))
                if 0 < j < len(chapters)
            )
            book.append(
                f'<section class="book-chapter" data-chapter="{slug[:2]}">'
                f'<header class="chapterhead" id="ch-{slug[:2]}"><div class="eyebrow">第 {i:02d} 课</div><h2>{escape(title)}</h2></header>'
                + contents(ch, "")
                + f'<nav class="adjacent" aria-label="分课阅读">{neighbors}</nav></section>'
            )
    files["handbook.html"] = reader(
        "知识点导航",
        "全课程合订本",
        "从目录选择课程或知识点。读到课末可翻到下一课。",
        "ai-infra-course-handbook",
        course_navigation(groups, chapters[1], handbook=True),
        "".join(book),
        "",
    )
    home = (HERE / "home.html").read_text("utf8")

    def row(ch):
        return f'<a class="lessonrow" href="{chapter_url(ch)}"><span class="num">{ch.get("number")}</span><div><h3>{escape(ch.get("title"))}</h3><p>{escape(text(ch.find("lead")))}</p></div><span class="go" aria-hidden="true">↗</span></a>'

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
    appearance = (HERE / "appearance.html").read_text("utf8").strip()
    for name, page in files.items():
        page = page.replace("{{APPEARANCE}}", appearance)
        if name in ("roadmap.html", "handbook.html"):
            page = page.replace(f'<a href="{name}">', f'<a aria-current="page" href="{name}">', 1)
        for asset in ("course.css", "course.js", "appearance.js", "index-data.js"):
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
