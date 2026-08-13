"""Render the editable revised-paper Markdown source as a PDF.

The renderer intentionally supports the small Markdown subset used by
``RIS_revised_paper.md`` so the generated PDF is reproducible without a TeX
installation or online service.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    # ReportLab paragraphs use a small XML-like markup language.
    escaped = html.escape(text, quote=False)
    escaped = escaped.replace("`", "")
    return Paragraph(escaped, style)


def parse_table(lines: List[str], start: int, style: ParagraphStyle):
    rows = []
    index = start
    while index < len(lines) and lines[index].startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(cell.replace("-", "").replace(":", "").strip() == "" for cell in cells):
            rows.append([paragraph(cell, style) for cell in cells])
        index += 1
    widths = [15.2 * cm / len(rows[0])] * len(rows[0])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b7c9d6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fbff")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table, index


def render(source: Path, output: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=10)
    author = ParagraphStyle("Author", parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=18)
    heading1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, spaceBefore=12, spaceAfter=6)
    heading2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=9, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=6)
    table_text = ParagraphStyle("TableText", parent=body, fontSize=7.2, leading=8.5, spaceAfter=0)
    code = ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=8.2, leading=10, leftIndent=12, backColor=colors.HexColor("#f2f2f2"), borderPadding=5, spaceBefore=3, spaceAfter=7)

    story = []
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    author_lines = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(paragraph(line[2:], title))
        elif line.startswith("## "):
            story.append(paragraph(line[3:], heading1))
        elif line.startswith("### "):
            story.append(paragraph(line[4:], heading2))
        elif line.startswith("|"):
            table, index = parse_table(lines, index, table_text)
            story.append(table)
            story.append(Spacer(1, 8))
            continue
        elif line.startswith("    "):
            story.append(paragraph(line[4:], code))
        elif line.startswith("- "):
            story.append(paragraph("- " + line[2:], body))
        elif author_lines < 2 and not story[-1:] == []:
            story.append(paragraph(line, author))
            author_lines += 1
        else:
            # Join a Markdown paragraph whose source was manually wrapped.
            paragraph_lines = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "- ", "    ")):
                paragraph_lines.append(lines[index].strip())
                index += 1
            story.append(paragraph(" ".join(paragraph_lines), body))
            continue
        index += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm,
        topMargin=1.55 * cm, bottomMargin=1.55 * cm, title="Relational Identity Structure (RIS)",
        author="Antonio Fallea",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(1.6 * cm, 0.85 * cm, "RIS reproducibility correction")
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.85 * cm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "paper" / "RIS_revised_paper.md")
    parser.add_argument("--output", type=Path, default=ROOT / "RIS.pdf")
    args = parser.parse_args()
    render(args.source, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
