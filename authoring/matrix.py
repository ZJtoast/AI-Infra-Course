"""Compute the illustrated causal tile from the same Q, K and V as the browser."""

import json
import math
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAME = "attention-tile-walkthrough.png"


def operands(lab):
    mats = [json.loads(lab.get("data-" + key)) for key in ("q", "k", "v")]
    if not all(
        len(m) == 4
        and all(
            len(row) == 2
            and all(isinstance(x, (float, int)) and math.isfinite(x) for x in row)
            for row in m
        )
        for m in mats
    ):
        raise ValueError(
            "The four-position attention lab expects three finite 4 × 2 matrices"
        )
    return mats


def snapshot(lab):
    q, k, v = operands(lab)
    scores = [
        [sum(a * b for a, b in zip(row, key)) / math.sqrt(2) for key in k] for row in q
    ]
    weights = []
    contributions = []
    outputs = []
    for i in (2, 3):
        maximum = max(scores[i][: i + 1])
        row = [math.exp(s - maximum) if j <= i else 0 for j, s in enumerate(scores[i])]
        weights.append(row[2:])
        contributions.append([sum(row[j] * v[j][d] for j in (2, 3)) for d in (0, 1)])
        outputs.append(
            [sum(w * vec[d] for w, vec in zip(row, v)) / sum(row) for d in (0, 1)]
        )
    scale = 3
    image = Image.new("RGB", (680 * scale, 412 * scale), "white")
    draw = ImageDraw.Draw(image)

    @lru_cache(None)
    def font(size, latin, bold):
        root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        name = (
            ("timesbd.ttf" if bold else "times.ttf")
            if latin
            else ("msyhbd.ttc" if bold else "msyh.ttc")
        )
        fallback = os.environ.get("COURSE_LATIN_FONT" if latin else "COURSE_CJK_FONT")
        return ImageFont.truetype(fallback or str(root / name), size * scale)

    def label(x, y, s, size=17, bold=False, anchor="mm"):
        s = str(s)
        draw.text(
            (x * scale, y * scale),
            s,
            anchor=anchor,
            fill="#19373F",
            font=font(size, s.isascii(), bold),
        )

    def grid(x, y, values, width, height, shade):
        for r, row in enumerate(values):
            for c, value in enumerate(row):
                left = x + c * width
                top = y + r * height
                draw.rectangle(
                    (
                        left * scale,
                        top * scale,
                        (left + width) * scale,
                        (top + height) * scale,
                    ),
                    fill=shade(r, c),
                    outline="#92aaa7",
                    width=scale,
                )
                s = (
                    format(value, ".2f").rstrip("0").rstrip(".")
                    if isinstance(value, float)
                    else str(value)
                )
                label(left + width / 2, top + height / 2, s, 16)

    label(20, 23, "在分数矩阵中追踪一个 2 × 2 块", 22, True, "lm")
    label(
        20, 53, "绿色为已处理的历史，橙色为当前块，叉号表示因果掩码。", 15, False, "lm"
    )
    for i in range(4):
        label(91 + 50 * i, 85, "k" + str(i), 16)
        label(42, 112 + 28 * i, "q" + str(i), 16)
    cells = [
        ["×" if j > i else scores[i][j] if i >= 2 else "·" for j in range(4)]
        for i in range(4)
    ]
    grid(
        66,
        98,
        cells,
        50,
        28,
        lambda r, c: "#f1f3f1" if r < 2 or c > r else "#c9e7df" if c < 2 else "#f9dfb6",
    )
    label(336, 152, "逐行取指数 →", 17)
    for i in range(2):
        label(460 + 80 * i, 94, "k" + str(i + 2), 16)
        label(397, 129 + 33 * i, "q" + str(i + 2), 16)
    grid(420, 113, weights, 80, 33, lambda *_: "#f9dfb6")
    label(500, 200, "当前块的指数权重 E", 16)
    for x, name, rows, color in [
        (41, "E · 当前权重块", weights, "#f9dfb6"),
        (245, "V · 第 2、3 行", v[2:], "#d9e8f7"),
        (452, "E V · 本块贡献", contributions, "#c9e7df"),
    ]:
        label(x + 68, 247, name, 16, True)
        grid(x, 267, rows, 68, 30, lambda *_, fill=color: fill)
    label(209, 298, "×", 22)
    label(419, 298, "=", 22)
    label(
        338,
        359,
        "合并旧块后，O 的第 2、3 行分别为 "
        + " 和 ".join("[" + ", ".join(f"{x:.2f}" for x in row) + "]" for row in outputs)
        + "。",
        16,
    )
    label(338, 390, "输出块占 O 的第 2、3 行，每一列对应 V 的一个通道。", 16)
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
