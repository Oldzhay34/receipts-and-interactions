"""Build the Phase 1 report (.docx) from the Markdown sections in ``reports/sections``.

What it does, in one pass over the sections (file name order ``01_...md``, ``02_...md``, ...):

* headings are numbered (``1``, ``1.1``, ``1.1.1``) and listed in a table of contents;
* ``{{sec:<key>}}`` becomes "Section N" (``SECTION_KEYS`` maps keys to section files);
* ``{{fig:<name>}}`` becomes "Figure N" and the image ``<name>.png`` is embedded once, after the
  block that first refers to it (looked up in the LATEST version of every stage under
  ``--outputs``);
* ``{{tab:<name>}}`` becomes "Table N". An inline Markdown table is bound to a name when the source
  comment right after it points to ``<name>.parquet`` (stage prefixes such as ``outliers_`` are
  allowed); names that are referenced but have no inline table are listed in an appendix with the
  file they come from;
* ``<!-- src: ... -->`` comments are removed from the document and written to ``sources.csv``
  (section, line, source, context);
* a cover page, page numbers in the footer and an estimate of the page count are added.

Usage::

    python reports/build_report.py --outputs <DATA_ROOT>/outputs \
        [--sections reports/sections] [--out reports/Phase1_Report.docx] \
        [--sources reports/Phase1_Report_sources.csv] [--authors "Name 1" "Name 2"]

Word fills in the table of contents when the document is opened (fields are marked for update).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

TITLE = "Ingredient-Based Recipe Recommendation with k-Nearest Neighbours"
SUBTITLE = "Phase 1 Report — Data Preparation, Exploration and a kNN Prototype"
COURSE = "CME 4403 Introduction to Machine Learning"
REPOSITORY = "https://github.com/Oldzhay34/receipts-and-interactions"

#: ``{{sec:<key>}}`` -> section file prefix.
SECTION_KEYS: dict[str, str] = {
    "introduction": "01",
    "dataset": "02",
    "exploration": "03",
    "quality": "04",
    "missing": "05",
    "outliers": "06",
    "ingredients": "07",
    "visualization": "08",
    "binning": "09",
    "scaling": "10",
    "sampling": "11",
    "knn": "12",
    "next_steps": "13",
}

#: Spelling unified at build time (the sections use British spelling; two headings did not).
TERM_FIXES: tuple[tuple[str, str], ...] = (
    (r"\bVisualizations\b", "Visualisations"),
    (r"\bNormalization\b", "Normalisation"),
)

REF_RE = re.compile(r"\{\{(fig|tab|sec):([A-Za-z0-9_\-]+)\}\}")
SRC_RE = re.compile(r"<!--\s*src:(.*?)-->", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*)")
LIST_RE = re.compile(r"^(\s*)([*\-]|\d+\.)\s+(.*)$")

# page estimate (Letter/A4, 1" margins, 11 pt): inches of usable height per page
PAGE_HEIGHT_IN = 9.0
#: measured / estimated pages for the 2026-11-23 build opened in Word 2016 (25 / 27.4)
CALIBRATION = 0.91
LINE_IN = 0.18
CHARS_PER_LINE = 105
BODY_PT = 10.5
#: image widths in inches: wide plots (histogram + box plot) and near-square plots
WIDE_IN, SQUARE_IN = 5.2, 3.6


@dataclass
class Block:
    kind: str  # heading | para | list | table
    text: str = ""
    level: int = 0
    items: list[tuple[bool, str]] = field(default_factory=list)  # (numbered, text)
    rows: list[list[str]] = field(default_factory=list)
    line: int = 0
    table_source: str | None = None  # src comment right after a table


@dataclass
class Section:
    path: Path
    number: int
    blocks: list[Block]


# --------------------------------------------------------------------------- parsing


def read_sections(folder: Path) -> list[Section]:
    files = sorted(p for p in folder.glob("*.md") if re.match(r"^\d\d_", p.name))
    if not files:
        raise SystemExit(f"no NN_*.md section files in {folder}")
    return [
        Section(p, i + 1, parse_markdown(p.read_text(encoding="utf-8")))
        for i, p in enumerate(files)
    ]


def parse_markdown(text: str) -> list[Block]:
    """Headings, paragraphs, bullet / numbered lists and pipe tables (enough for the sections)."""
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        start = i + 1
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(Block("heading", line[level:].strip(), level=level, line=start))
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", rows=rows, line=start))
        elif line.strip().startswith("<!--") and blocks and blocks[-1].kind == "table":
            chunk = [line]
            while "-->" not in chunk[-1] and i + 1 < len(lines):
                i += 1
                chunk.append(lines[i])
            blocks[-1].table_source = "\n".join(chunk)
            i += 1
        elif LIST_RE.match(line):
            items: list[tuple[bool, str]] = []
            while i < len(lines) and lines[i].strip():
                m = LIST_RE.match(lines[i])
                if m and len(m.group(1)) < 2:
                    items.append((m.group(2)[0].isdigit(), m.group(3).strip()))
                elif items:  # continuation line of the current item
                    items[-1] = (items[-1][0], items[-1][1] + " " + lines[i].strip())
                i += 1
            blocks.append(Block("list", items=items, line=start))
        else:
            para = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].startswith("#")
                and not (lines[i].lstrip().startswith("|"))
                and not (para and LIST_RE.match(lines[i]))
            ):
                para.append(lines[i].strip())
                i += 1
            blocks.append(Block("para", " ".join(para), line=start))
    return blocks


# --------------------------------------------------------------------------- references


class Registry:
    """Numbers figures, tables and sections in order of first reference."""

    def __init__(self, sections: list[Section], outputs: Path | None) -> None:
        self.section_no = {
            key: next((s.number for s in sections if s.path.name.startswith(prefix)), None)
            for key, prefix in SECTION_KEYS.items()
        }
        self.figures: dict[str, int] = {}
        self.tables: dict[str, int] = {}
        self.images = _index_files(outputs, ".png")
        self.parquets = _index_files(outputs, ".parquet")
        self.problems: list[str] = []
        self.placed_tables: set[str] = set()

    def number(self, kind: str, name: str) -> int:
        store = self.figures if kind == "fig" else self.tables
        if name not in store:
            store[name] = len(store) + 1
        return store[name]

    def resolve(self, text: str, where: str) -> str:
        for pattern, replacement in TERM_FIXES:
            text = re.sub(pattern, replacement, text)

        def repl(m: re.Match[str]) -> str:
            kind, name = m.group(1), m.group(2)
            if kind == "sec":
                no = self.section_no.get(name)
                if no is None:
                    self.problems.append(f"{where}: unknown section key {name!r}")
                    return "Section ?"
                return f"Section {no}"
            return f"{'Figure' if kind == 'fig' else 'Table'} {self.number(kind, name)}"

        return REF_RE.sub(repl, text)

    def table_file(self, name: str) -> Path | None:
        """``<name>.parquet``, or ``<stage>/<rest>.parquet`` for names like ``outliers_iqr``."""
        if name in self.parquets:
            return self.parquets[name]
        for stem, path in self.parquets.items():
            if name == f"{path.parent.parent.name}_{stem}":
                return path
        return None


def _index_files(outputs: Path | None, suffix: str) -> dict[str, Path]:
    """name -> file, over the LATEST version of every stage under ``outputs``."""
    found: dict[str, Path] = {}
    if outputs is None:
        return found
    for latest in sorted(outputs.glob("*/LATEST")):
        version = latest.read_text(encoding="utf-8").strip()
        for f in sorted((latest.parent / version).glob(f"*{suffix}")):
            found.setdefault(f.stem, f)
    return found


def caption_from_name(name: str) -> str:
    text = re.sub(r"^fig_\d+_\d+_", "", name).replace("_", " ")
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- docx helpers


def add_field(paragraph, instruction: str) -> None:  # noqa: ANN001 - python-docx paragraph
    run = paragraph.add_run()
    for tag, text in (("begin", None), (None, instruction), ("separate", None), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def add_inline(paragraph, text: str) -> None:  # noqa: ANN001
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def caption(doc, label: str, text: str) -> None:  # noqa: ANN001
    p = doc.add_paragraph(style="Caption")
    p.add_run(f"{label}. ").bold = True
    p.add_run(text)


# --------------------------------------------------------------------------- build


@dataclass
class BuildResult:
    sources: list[dict[str, str]]
    problems: list[str]
    est_pages: float
    n_figures: int
    n_tables: int


def build(
    sections_dir: Path,
    outputs: Path | None,
    out_docx: Path,
    authors: Iterable[str],
    date: str,
) -> BuildResult:
    sections = read_sections(sections_dir)
    reg = Registry(sections, outputs)
    doc = Document()
    _base_styles(doc)
    height = _cover(doc, authors, date) + _toc(doc)
    sources: list[dict[str, str]] = []

    for sec in sections:
        counters = [sec.number, 0, 0]
        where = sec.path.name
        for block in sec.blocks:
            raw = block.text + " ".join(t for _, t in block.items)
            raw += " ".join(" ".join(r) for r in block.rows) + (block.table_source or "")
            for m in SRC_RE.finditer(raw):
                context = COMMENT_RE.sub("", raw)
                sources.append(
                    {
                        "section": where,
                        "line": str(block.line),
                        "source": " ".join(m.group(1).split()),
                        "context": " ".join(context.split())[:240],
                    }
                )
            figs_before = set(reg.figures)
            if block.kind == "heading":
                level = min(block.level, 3)
                counters[level - 1] = counters[level - 1] + (1 if level > 1 else 0)
                for k in range(level, 3):
                    counters[k] = 0
                number = ".".join(str(c) for c in counters[:level])
                title = reg.resolve(COMMENT_RE.sub("", block.text), where)
                doc.add_heading(f"{number} {title}", level=level)
                height += 0.5
            elif block.kind == "para":
                text = reg.resolve(COMMENT_RE.sub("", block.text), where).strip()
                if text:
                    add_inline(doc.add_paragraph(), " ".join(text.split()))
                    height += _text_height(text) + 0.1
            elif block.kind == "list":
                for numbered, item in block.items:
                    text = reg.resolve(COMMENT_RE.sub("", item), where)
                    style = "List Number" if numbered else "List Bullet"
                    add_inline(doc.add_paragraph(style=style), " ".join(text.split()))
                    height += _text_height(text, indent=True)
                height += 0.1
            elif block.kind == "table":
                height += _table(doc, block, reg, where)
            new_figs = [f for f in reg.figures if f not in figs_before]
            for name in new_figs:
                height += _figure(doc, name, reg, where)

    height += _appendix(doc, reg)
    _footer_page_numbers(doc)
    _update_fields_on_open(doc)
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_docx)
    return BuildResult(
        sources,
        reg.problems,
        CALIBRATION * height / PAGE_HEIGHT_IN,
        len(reg.figures),
        len(reg.tables),
    )


def _base_styles(doc) -> None:  # noqa: ANN001
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.0
    for name in ("List Bullet", "List Number", "Caption"):
        doc.styles[name].paragraph_format.space_after = Pt(2)
    for s in doc.sections:
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)


def _cover(doc, authors: Iterable[str], date: str) -> float:  # noqa: ANN001
    for _ in range(6):
        doc.add_paragraph()
    for text, size, bold in ((TITLE, 24, True), (SUBTITLE, 14, False), (COURSE, 12, False)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
    doc.add_paragraph()
    for text in (*authors, date, REPOSITORY):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(text)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    return PAGE_HEIGHT_IN


def _toc(doc) -> float:  # noqa: ANN001
    doc.add_paragraph().add_run("Contents").bold = True
    add_field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    return PAGE_HEIGHT_IN


def _text_height(text: str, indent: bool = False) -> float:
    per_line = CHARS_PER_LINE - (8 if indent else 0)
    return max(1, -(-len(text) // per_line)) * LINE_IN


def _table(doc, block: Block, reg: Registry, where: str) -> float:  # noqa: ANN001
    name = _bound_table_name(block, reg)
    if name:
        reg.placed_tables.add(name)
        label = f"Table {reg.number('tab', name)}"
        title = caption_from_name(name)
    else:
        label = f"Table {reg.number('tab', f'_inline_{where}_{block.line}')}"
        title = "Summary"
    rows = [[reg.resolve(COMMENT_RE.sub("", c), where) for c in r] for r in block.rows]
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c in range(ncols):
            cell = table.cell(r, c)
            cell.text = ""
            par = cell.paragraphs[0]
            add_inline(par, row[c] if c < len(row) else "")
            for run in par.runs:
                run.font.size = Pt(8.5)
                if r == 0:
                    run.bold = True
    caption(doc, label, title)
    return 0.3 + sum(max(_text_height(" ".join(r)) * 0.6, 0.25) for r in rows)


def _bound_table_name(block: Block, reg: Registry) -> str | None:
    """The {{tab:...}} name whose parquet file is cited in the comment after the table."""
    if not block.table_source:
        return None
    m = re.search(r"/([A-Za-z0-9_]+)\.parquet", block.table_source)
    if not m:
        return None
    stem = m.group(1)
    for name in reg.tables:
        f = reg.table_file(name)
        if f is not None and f.stem == stem and name not in reg.placed_tables:
            return name
    if stem in reg.parquets and stem not in reg.placed_tables:
        return stem
    return None


def _figure(doc, name: str, reg: Registry, where: str) -> float:  # noqa: ANN001
    label = f"Figure {reg.figures[name]}"
    path = reg.images.get(name)
    if path is None:
        reg.problems.append(f"{where}: image {name}.png not found under --outputs")
        caption(doc, label, f"{caption_from_name(name)} [image {name}.png not found]")
        return 0.4
    with Image.open(path) as im:
        w, h = im.size
    width = WIDE_IN if w / h >= 1.6 else SQUARE_IN
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, label, caption_from_name(name))
    return width * h / w + 0.4


def _appendix(doc, reg: Registry) -> float:  # noqa: ANN001
    missing = [n for n in reg.tables if not n.startswith("_inline_") and n not in reg.placed_tables]
    if not missing:
        return 0.0
    doc.add_heading("Appendix: referenced data tables", level=1)
    doc.add_paragraph(
        "These tables are referenced in the text but too large to print in full; each is "
        "delivered with the submission as the file named here (latest version of its stage)."
    )
    height = 1.0
    for name in missing:
        f = reg.table_file(name)
        where = f"{f.parent.parent.name}/{f.parent.name}/{f.name}" if f else "file not found"
        if f is None:
            reg.problems.append(f"table {name!r}: no {name}.parquet under --outputs")
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"Table {reg.tables[name]}").bold = True
        p.add_run(f" — {caption_from_name(name)}: outputs/{where}")
        height += LINE_IN
    return height


def _footer_page_numbers(doc) -> None:  # noqa: ANN001
    for section in doc.sections:
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(p, "PAGE")


def _update_fields_on_open(doc) -> None:  # noqa: ANN001
    settings = doc.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)


def write_sources(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "line", "source", "context"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", type=Path, default=REPO_ROOT / "reports" / "sections")
    ap.add_argument("--outputs", type=Path, help="outputs/ folder (DATA_ROOT/outputs) for figures")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "Phase1_Report.docx")
    ap.add_argument("--sources", type=Path, default=None, help="default: <out>_sources.csv")
    ap.add_argument("--authors", nargs="*", default=["Person A", "Person B"])
    ap.add_argument("--date", default=dt.date.today().strftime("%d %B %Y"))
    ap.add_argument("--strict", action="store_true", help="exit 1 if any reference is unresolved")
    args = ap.parse_args(argv)
    sources_path = args.sources or args.out.with_name(args.out.stem + "_sources.csv")

    res = build(args.sections, args.outputs, args.out, args.authors, args.date)
    write_sources(res.sources, sources_path)
    print(f"wrote {args.out} ({res.n_figures} figures, {res.n_tables} tables)")
    print(f"wrote {sources_path} ({len(res.sources)} source comments)")
    print(f"estimated length: {res.est_pages:.1f} pages (incl. cover and contents)")
    for p in res.problems:
        print("WARNING:", p)
    return 1 if (args.strict and res.problems) else 0


if __name__ == "__main__":
    sys.exit(main())
