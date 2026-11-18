"""Place the operands of an annotated product and their shapes on two shared baselines."""

from lxml import etree as ET

MATH = "{http://www.w3.org/1998/Math/MathML}"


def dimensioned(node):
    return (
        node.tag == MATH + "munder"
        and len(node) == 2
        and node[0].tag == MATH + "munder"
        and len(node[0]) == 2
        and node[0][1].text == "⏟"
    )


def aligned_dimensions(math):
    annotated = [node for node in math.iter(MATH + "munder") if dimensioned(node)]
    if not annotated:
        return
    # Split grouping rows only. Fractions, subscripts and matrices remain intact.
    paths = {parent for node in annotated for parent in node.iterancestors() if parent is not math and parent.tag.startswith(MATH)}
    if any(node.tag != MATH + "mrow" for node in paths):
        return

    def cells(row):
        for item in list(row):
            if item in paths:
                yield from cells(item)
            else:
                yield item

    grid = ET.Element(MATH + "mtable", {
        "class": "dimension-grid", "rowalign": "baseline", "columnalign": "center",
        "columnspacing": "0.12em", "rowspacing": "0.14em",
    })
    terms = ET.SubElement(grid, MATH + "mtr")
    shapes = ET.SubElement(grid, MATH + "mtr")
    plain = None
    for item in list(cells(math)):
        if dimensioned(item):
            ET.SubElement(terms, MATH + "mtd").append(item[0][0])
            label = ET.SubElement(shapes, MATH + "mtd", {"class": "dimension"})
            ET.SubElement(label, MATH + "mstyle", scriptlevel="+1").append(item[1])
            plain = None
        else:
            if plain is None:
                plain = ET.SubElement(ET.SubElement(terms, MATH + "mtd"), MATH + "mrow")
                ET.SubElement(shapes, MATH + "mtd")
            plain.append(item)
    math[:] = [grid]
