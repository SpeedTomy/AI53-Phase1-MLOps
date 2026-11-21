"""Render LAB_REPORT.md with explicit page breaks into the hand-in PDF.

Optional document dependencies are separate from the locked ML environment:
uv run --no-project --python 3.12 --with reportlab==5.1.0 --with markdown==3.11.3 \
    python scripts/build_report.py
"""
from __future__ import annotations

import html
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/SpeedTomy/AI53-Phase1-MLOps"
WIDTH = A4[0] - 88
NAVY = colors.HexColor("#173A56")
INK = colors.HexColor("#243442")
MUTED = colors.HexColor("#586875")
LINE = colors.HexColor("#D8E0E6")
PALE = colors.HexColor("#F0F4F7")


def fonts() -> tuple[str, str]:
    mac = Path("/System/Library/Fonts/Supplemental")
    linux = Path("/usr/share/fonts/truetype/dejavu")
    candidates = [
        (mac / "Arial.ttf", mac / "Arial Bold.ttf", mac / "Arial Italic.ttf",
         mac / "Arial Bold Italic.ttf", mac / "Courier New.ttf"),
        (linux / "DejaVuSans.ttf", linux / "DejaVuSans-Bold.ttf",
         linux / "DejaVuSans-Oblique.ttf", linux / "DejaVuSans-BoldOblique.ttf",
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")),
    ]
    for regular, bold, italic, bold_italic, mono in candidates:
        if all(p.exists() for p in (regular, bold, italic, bold_italic, mono)):
            for name, path in (("ReportSans", regular), ("ReportSans-Bold", bold),
                               ("ReportSans-Italic", italic),
                               ("ReportSans-BoldItalic", bold_italic), ("ReportMono", mono)):
                pdfmetrics.registerFont(TTFont(name, str(path)))
            pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans",
                bold="ReportSans-Bold", italic="ReportSans-Italic",
                boldItalic="ReportSans-BoldItalic")
            return "ReportSans", "ReportMono"
    return "Helvetica", "Courier"


SANS, MONO = fonts()
STYLES = {
    "body": ParagraphStyle("body", fontName=SANS, fontSize=9.6, leading=13.2,
        textColor=INK, spaceAfter=7, splitLongWords=True),
    "h1": ParagraphStyle("h1", fontName=SANS, fontSize=23, leading=28,
        textColor=NAVY, spaceAfter=13, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName=SANS, fontSize=15.5, leading=20,
        textColor=NAVY, spaceAfter=10, spaceBefore=1, keepWithNext=True),
    "h3": ParagraphStyle("h3", fontName=SANS, fontSize=11.3, leading=15,
        textColor=NAVY, spaceBefore=9, spaceAfter=5, keepWithNext=True),
    "cell": ParagraphStyle("cell", fontName=SANS, fontSize=8.3, leading=11,
        textColor=INK, splitLongWords=True),
    "caption": ParagraphStyle("caption", fontName=SANS, fontSize=8.3, leading=11,
        textColor=MUTED, spaceBefore=2, spaceAfter=8),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=7.9, leading=10.5,
        textColor=NAVY, backColor=PALE, borderPadding=7, spaceBefore=3, spaceAfter=9),
}


def inline(node: ET.Element) -> str:
    result = html.escape(node.text or "", quote=False)
    for child in node:
        content = inline(child)
        tag = child.tag
        if tag == "strong":
            result += f"<b>{content}</b>"
        elif tag == "em":
            result += f"<i>{content}</i>"
        elif tag == "code":
            result += f'<font face="{MONO}" size="7.9">{content}</font>'
        elif tag == "a":
            href = child.attrib.get("href", "")
            if not re.match(r"[a-z]+:", href):
                href = REPOSITORY + "/blob/main/" + href
            result += f'<a href="{html.escape(href, quote=True)}" color="#174A73">{content}</a>'
        elif tag == "br":
            result += "<br/>"
        else:
            result += content
        result += html.escape(child.tail or "", quote=False)
    return result


def table_widths(headers: list[str]) -> list[float]:
    first = headers[0]
    count = len(headers)
    if first == "Ensemble":
        ratios = [0.24, 0.14, 0.62]
    elif first == "Élément":
        ratios = [0.24, 0.34, 0.42]
    elif first == "Classe":
        ratios = [0.43, 0.19, 0.19, 0.19]
    elif first == "Étape":
        ratios = [0.17, 0.41, 0.42]
    elif count == 5:
        ratios = [0.36, 0.16, 0.16, 0.14, 0.18]
    elif count == 6:
        ratios = [0.07, 0.25, 0.18, 0.18, 0.16, 0.16]
    elif first == "Run" and count == 3:
        ratios = [0.08, 0.56, 0.36]
    elif first == "Preuve":
        ratios = [0.36, 0.64]
    elif count == 2:
        ratios = [0.68, 0.32]
    else:
        ratios = [1 / count] * count
    return [WIDTH * ratio for ratio in ratios]


def flowables(fragment: str) -> list:
    tree = ET.fromstring("<root>" + markdown.markdown(fragment,
        extensions=["tables", "fenced_code"]) + "</root>")
    story = []
    for node in tree:
        if node.tag in ("h1", "h2", "h3"):
            story.append(Paragraph(inline(node), STYLES[node.tag]))
        elif node.tag == "p":
            image = node.find("img")
            if image is not None:
                figure = Image(str(ROOT / image.attrib["src"]))
                figure.drawHeight *= WIDTH / figure.drawWidth
                figure.drawWidth = WIDTH
                story += [Spacer(1, 4), figure, Spacer(1, 3)]
            else:
                style = "caption" if node.find("em") is not None and (node.text or "").strip() == "" else "body"
                story.append(Paragraph(inline(node), STYLES[style]))
        elif node.tag == "pre":
            story.append(Preformatted("".join(node.itertext()).rstrip(), STYLES["code"],
                                      maxLineLength=105))
        elif node.tag == "table":
            rows = list(node.findall("./thead/tr")) + list(node.findall("./tbody/tr"))
            headers = ["".join(cell.itertext()) for cell in rows[0]]
            data = [[Paragraph(("<b>" + inline(cell) + "</b>") if i == 0 else inline(cell),
                               STYLES["cell"]) for cell in row] for i, row in enumerate(rows)]
            table = Table(data, colWidths=table_widths(headers), repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 9)])
        elif node.tag in ("ul", "ol"):
            for i, item in enumerate(node):
                prefix = "- " if node.tag == "ul" else f"{i + 1}. "
                story.append(Paragraph(prefix + inline(item), STYLES["body"]))
    return story


def decorate(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont(SANS, 8)
    canvas.drawString(44, A4[1] - 30, "AI53 | Projet MLOps - Phase 1")
    canvas.drawRightString(A4[0] - 44, A4[1] - 30, "Rapport de laboratoire")
    canvas.setStrokeColor(LINE)
    canvas.line(44, 38, A4[0] - 44, 38)
    canvas.setFont(SANS, 7.7)
    canvas.drawString(44, 25, "Mansour Thomas Yazan | Novembre 2026")
    canvas.drawRightString(A4[0] - 44, 25, str(doc.page))
    canvas.restoreState()


def main() -> None:
    output = ROOT / "output/pdf/AI53_Phase1_Rapport.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4,
        leftMargin=44, rightMargin=44, topMargin=49, bottomMargin=50,
        title="AI53 - Pipeline MLOps reproductible - Phase 1",
        author="Mansour Thomas Yazan", subject="Rapport de laboratoire AI53")
    sections = (ROOT / "LAB_REPORT.md").read_text(encoding="utf-8").split("<!-- pagebreak -->")
    story = []
    for index, section in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.extend(flowables(section))
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
