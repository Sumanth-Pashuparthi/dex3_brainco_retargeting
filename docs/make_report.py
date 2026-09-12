#!/usr/bin/env python3
"""Render assignment1_report.md to a paginated PDF.

There is no pandoc or LaTeX on the machine this was written on, and adding either would make the
report un-buildable for anyone who clones the repo without them. matplotlib is already a dependency
of the analysis, so the report is typeset with that and nothing else.

The markdown subset supported is the one the report uses: headings, paragraphs with ``**bold**`` and
``` `code` ``` inline, ``-`` bullets, ``>`` pulled-out lines, pipe tables, ``![caption](path)``
figures and ``---`` page breaks. It is a renderer for this document, not a general markdown engine.

    python3 make_report.py                    # -> assignment1_report.pdf
    python3 make_report.py --out draft.pdf

Figure paths are resolved relative to this file, so the build is independent of where the
repository lives and of the working directory it is run from.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent

# A4. Margins, type sizes and FIG_SCALE are together sized to land the whole report in four pages.
PAGE_W, PAGE_H = 8.27, 11.69
MARGIN_L, MARGIN_R = 0.58, 0.58
MARGIN_T, MARGIN_B = 0.52, 0.45
BODY_W = PAGE_W - MARGIN_L - MARGIN_R

# Every mark that carries type or rules is pure black on white: the report is printed flat, and
# nothing in the argument depends on a colour being reproduced. The figures are the only colour in
# the document, because there hue carries real information. Hierarchy is set with weight, size and
# rule thickness rather than with grey.
SAVE_DPI = 300       # resolution of embedded bitmaps only; all type is vector
FONT = "DejaVu Sans"
SZ_TITLE, SZ_H2, SZ_H3, SZ_BODY, SZ_CAPTION, SZ_TABLE = 14.0, 10.0, 9.0, 7.9, 6.9, 7.2
LEAD = 1.17          # line spacing, multiples of font size
INK = "#000000"
RULE = "#000000"     # table and heading rules; weight comes from lw, not from lightening the ink

# Figures are inset to this fraction of the body width. Lower means more text per page.
FIG_SCALE = 0.65


class Page:
    """One PDF page, with a cursor that moves down the page in inches from the top."""

    def __init__(self, pdf: PdfPages, number: int):
        self.pdf = pdf
        self.number = number
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.fig.patch.set_facecolor("white")
        self.y = MARGIN_T
        self._renderer = self.fig.canvas.get_renderer()

    # inches -> figure fraction
    def fx(self, x_in: float) -> float:
        return x_in / PAGE_W

    def fy(self, y_in: float) -> float:
        return 1.0 - y_in / PAGE_H

    def room(self, need_in: float) -> bool:
        return self.y + need_in <= PAGE_H - MARGIN_B

    def text(self, x_in, s, size, *, weight="normal", color=INK, family=FONT):
        return self.fig.text(self.fx(x_in), self.fy(self.y), s, fontsize=size, color=color,
                             fontweight=weight, family=family, va="top", ha="left")

    def width_of(self, s, size, weight="normal", family=FONT) -> float:
        """Rendered width of a string, in inches."""
        t = self.fig.text(0, 0, s, fontsize=size, fontweight=weight, family=family)
        w = t.get_window_extent(renderer=self._renderer).width / self.fig.dpi
        t.remove()
        return w

    def close(self):
        self.fig.text(self.fx(PAGE_W / 2 - MARGIN_L), self.fy(PAGE_H - MARGIN_B + 0.34),
                      str(self.number), fontsize=7.0, color=INK, family=FONT, ha="center")
        # Text stays vector at any dpi; this only sets the resolution of embedded bitmaps.
        self.pdf.savefig(self.fig, dpi=SAVE_DPI)
        plt.close(self.fig)


class Doc:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.n = 0
        self.page = self._new()

    def _new(self) -> Page:
        self.n += 1
        return Page(self.pdf, self.n)

    def break_page(self):
        self.page.close()
        self.page = self._new()

    def need(self, inches: float):
        if not self.page.room(inches):
            self.break_page()

    def space(self, inches: float):
        self.page.y += inches


# ------------------------------------------------------------------ inline bold handling

MONO = "DejaVu Sans Mono"

# A token is one word plus how to draw it. `glue` means "no space before me", which is what keeps
# the full stop in "...recovers it to **0.95**." from drifting away from the bold run.
class Tok:
    __slots__ = ("text", "bold", "mono", "glue")

    def __init__(self, text: str, bold: bool, mono: bool, glue: bool):
        self.text, self.bold, self.mono, self.glue = text, bold, mono, glue

    @property
    def weight(self) -> str:
        return "bold" if self.bold else "normal"

    @property
    def family(self) -> str:
        return MONO if self.mono else FONT


_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def tokenize(s: str) -> list[Tok]:
    """Split a line of markdown into drawable word tokens.

    Handles ``**bold**`` and ``` `code` ``` spans. Whitespace between the source segments is what
    decides whether a token is glued to its predecessor, so punctuation that immediately follows a
    span stays attached to it.
    """
    toks: list[Tok] = []
    pending_glue = False  # the previous segment ended without whitespace
    for part in _INLINE.split(s):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            body, bold, mono = part[2:-2], True, False
        elif part.startswith("`") and part.endswith("`"):
            body, bold, mono = part[1:-1], False, True
        else:
            body, bold, mono = part, False, False

        starts_space = body[:1].isspace()
        words = body.split()
        for i, w in enumerate(words):
            glue = (i == 0) and pending_glue and not starts_space and bool(toks)
            toks.append(Tok(w, bold, mono, glue))
        if words:
            pending_glue = not body[-1:].isspace()
        elif body:
            # A whitespace-only segment carries no words but still separates the spans either side
            # of it, so it has to clear the flag -- otherwise "**a.** **b**" renders as "a.b".
            pending_glue = False
    return toks


# Code spans sit slightly smaller: DejaVu Mono runs wider than the body face at the same size.
def _fontsize(tok: Tok, size: float) -> float:
    return size * 0.94 if tok.mono else size


def tok_width(page: "Page", tok: Tok, size: float) -> float:
    """Width of one token, in inches. The single place token widths are measured, so that the
    wrap, the unwrapped-width and the draw pass can never disagree about where a line ends."""
    return page.width_of(tok.text, _fontsize(tok, size), tok.weight, tok.family)


def bolded(toks: list[Tok]) -> list[Tok]:
    """Force a whole run bold, for table headers."""
    return [Tok(t.text, True, t.mono, t.glue) for t in toks]


def line_width(doc: Doc, toks: list[Tok], size: float) -> float:
    """Width these tokens would occupy on one unwrapped line, in inches."""
    space_w = doc.page.width_of(" ", size)
    total = 0.0
    for i, tok in enumerate(toks):
        if i > 0 and not tok.glue:
            total += space_w
        total += tok_width(doc.page, tok, size)
    return total


def line_count(doc: Doc, toks: list[Tok], width_in: float, size: float) -> int:
    """How many lines these tokens will wrap to, without drawing anything."""
    space_w = doc.page.width_of(" ", size)
    n, cur = 1, 0.0
    for tok in toks:
        w = tok_width(doc.page, tok, size)
        gap = 0.0 if (tok.glue or cur == 0.0) else space_w
        if cur and cur + gap + w > width_in:
            n += 1
            cur = w
        else:
            cur += gap + w
    return n


def draw_wrapped(doc: Doc, toks: list[Tok], x_in: float, width_in: float,
                 size: float, *, color=INK) -> None:
    """Greedy word wrap using measured widths, so bold and code runs never overflow the margin."""
    space_w = doc.page.width_of(" ", size)
    line: list[tuple[Tok, float]] = []
    line_w = 0.0

    def flush():
        nonlocal line, line_w
        if not line:
            return
        doc.need(size * LEAD / 72)
        x = x_in
        for i, (tok, w) in enumerate(line):
            if i > 0 and not tok.glue:
                x += space_w
            doc.page.text(x, tok.text, _fontsize(tok, size),
                          weight=tok.weight, color=color, family=tok.family)
            x += w
        doc.page.y += size * LEAD / 72
        line, line_w = [], 0.0

    for tok in toks:
        w = tok_width(doc.page, tok, size)
        gap = 0.0 if (tok.glue or not line) else space_w
        if line and line_w + gap + w > width_in:
            flush()
            line, line_w = [(tok, w)], w   # a glued token starting a line loses its glue
        else:
            line.append((tok, w))
            line_w += gap + w
    flush()


# ------------------------------------------------------------------ block renderers

def render_heading(doc: Doc, level: int, text: str) -> None:
    if level == 1:
        doc.space(0.06)
        draw_wrapped(doc, tokenize(text), MARGIN_L, BODY_W, SZ_TITLE)
        doc.space(0.07)
        doc.page.fig.add_artist(plt.Line2D(
            [doc.page.fx(MARGIN_L), doc.page.fx(MARGIN_L + BODY_W)],
            [doc.page.fy(doc.page.y)] * 2, color=INK, lw=1.1))
        doc.space(0.14)
        return

    size = SZ_H2 if level == 2 else SZ_H3
    # Keep a heading with at least a couple of lines of what follows it.
    doc.need(size * LEAD / 72 + 0.44)
    doc.space(0.10 if level == 2 else 0.08)
    bold = [Tok(t.text, True, t.mono, t.glue) for t in tokenize(text)]
    draw_wrapped(doc, bold, MARGIN_L, BODY_W, size)
    if level == 2:
        doc.space(0.025)
        doc.page.fig.add_artist(plt.Line2D(
            [doc.page.fx(MARGIN_L), doc.page.fx(MARGIN_L + BODY_W)],
            [doc.page.fy(doc.page.y)] * 2, color=RULE, lw=0.4))
    doc.space(0.055)


def render_paragraph(doc: Doc, text: str) -> None:
    draw_wrapped(doc, tokenize(text), MARGIN_L, BODY_W, SZ_BODY)
    doc.space(0.055)


def render_quote(doc: Doc, text: str) -> None:
    """Pulled-out line, indented behind a rule. Without this the ``>`` reached the page as text."""
    indent = 0.22
    toks = tokenize(text)
    height = line_count(doc, toks, BODY_W - indent, SZ_BODY) * SZ_BODY * LEAD / 72
    doc.need(height + 0.10)
    doc.space(0.05)
    top = doc.page.y
    draw_wrapped(doc, toks, MARGIN_L + indent, BODY_W - indent, SZ_BODY)
    doc.page.fig.add_artist(plt.Line2D(
        [doc.page.fx(MARGIN_L + 0.02)] * 2,
        [doc.page.fy(top - 0.01), doc.page.fy(doc.page.y - 0.02)], color=RULE, lw=1.2))
    doc.space(0.08)


def render_bullets(doc: Doc, items: list[str]) -> None:
    indent = 0.19
    for item in items:
        doc.need(SZ_BODY * LEAD / 72)
        doc.page.text(MARGIN_L + 0.04, "\u2013", SZ_BODY, color=INK)
        draw_wrapped(doc, tokenize(item), MARGIN_L + indent,
                     BODY_W - indent, SZ_BODY)
        doc.space(0.03)
    doc.space(0.05)


def render_table(doc: Doc, rows: list[list[str]]) -> None:
    """Pipe table. First row is the header; an empty header cell is allowed."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]

    # Header cells are drawn bold, so tokenise once per cell with that already applied.
    toks = [[bolded(tokenize(cell)) if i == 0 else tokenize(cell) for cell in row]
            for i, row in enumerate(rows)]

    # Column widths from the unwrapped width of the content, then scaled to the body width.
    natural = []
    for c in range(ncol):
        w = max(line_width(doc, toks[i][c], SZ_TABLE) for i in range(len(rows)))
        natural.append(w + 0.18)
    total = sum(natural)
    widths = [w * BODY_W / total for w in natural] if total > BODY_W else natural

    line_h = SZ_TABLE * LEAD / 72
    heights = [max(line_count(doc, toks[i][c], widths[c] - 0.14, SZ_TABLE)
                   for c in range(ncol)) * line_h
               for i in range(len(rows))]

    doc.need(heights[0] + (heights[1] if len(heights) > 1 else 0) + 0.2)
    doc.space(0.06)

    for i, row in enumerate(rows):
        if i > 0 and not doc.page.room(heights[i] + 0.05):
            doc.break_page()
        top = doc.page.y
        x = MARGIN_L
        for c in range(ncol):
            saved = doc.page.y
            draw_wrapped(doc, toks[i][c], x + 0.05, widths[c] - 0.10, SZ_TABLE)
            doc.page.y = saved
            x += widths[c]
        doc.page.y = top + heights[i]
        # Header and body rules are the same ink; only the weight separates them.
        doc.page.fig.add_artist(plt.Line2D(
            [doc.page.fx(MARGIN_L), doc.page.fx(MARGIN_L + sum(widths))],
            [doc.page.fy(doc.page.y + 0.015)] * 2,
            color=RULE, lw=0.9 if i == 0 else 0.4))
        doc.space(0.055)
    doc.space(0.10)


def render_figure(doc: Doc, caption: str, rel_path: str) -> None:
    path = (HERE / rel_path).resolve()
    if not path.exists():
        render_paragraph(doc, f"[missing figure: {rel_path} -- run 5_gr00t_finetune/analyze.py]")
        return

    img = mpimg.imread(path)
    ih, iw = img.shape[0], img.shape[1]
    # Inset so figures read as figures rather than as full-bleed page furniture.
    w_in = BODY_W * FIG_SCALE
    h_in = w_in * ih / iw

    # Measure the caption instead of guessing at it. A fixed guess is what used to push a wrapped
    # caption past the bottom margin, and what made the renderer break a page it did not need to.
    cap_words = tokenize(caption)
    lead_w = doc.page.width_of(f"Figure {doc_fig_number(doc)}. ", SZ_CAPTION, "bold")
    cap_h = 0.0
    if cap_words:
        lines = line_count(doc, cap_words, BODY_W - lead_w, SZ_CAPTION)
        cap_h = 0.03 + lines * SZ_CAPTION * LEAD / 72

    if not doc.page.room(0.05 + h_in + cap_h + 0.10):
        doc.break_page()

    doc.space(0.05)
    ax = doc.page.fig.add_axes([
        doc.page.fx(MARGIN_L + (BODY_W - w_in) / 2),
        doc.page.fy(doc.page.y + h_in),
        w_in / PAGE_W,
        h_in / PAGE_H,
    ])
    # interpolation="none" matters: with any other value matplotlib resamples the bitmap down to
    # the on-page size at the figure's dpi before writing it, which is what made the figures look
    # soft. With "none" the PDF backend embeds the original pixels and the viewer scales them.
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    doc.page.y += h_in + 0.03

    if cap_words:
        doc.page.text(MARGIN_L, f"Figure {doc_fig_number(doc)}.", SZ_CAPTION,
                      weight="bold", color=INK)
        draw_wrapped(doc, cap_words, MARGIN_L + lead_w, BODY_W - lead_w, SZ_CAPTION,
                     color=INK)
    doc.space(0.10)


_FIG_COUNT = {"n": 0}


def doc_fig_number(doc: Doc) -> int:
    return _FIG_COUNT["n"]


# ------------------------------------------------------------------ markdown parsing

def parse_blocks(md: str) -> list[tuple]:
    """Markdown -> a list of ('kind', payload) blocks."""
    blocks: list[tuple] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line:
            i += 1
            continue

        if line == "---":
            blocks.append(("pagebreak", None))
            i += 1
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            blocks.append(("heading", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue

        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
        if m:
            blocks.append(("figure", (m.group(1), m.group(2))))
            i += 1
            continue

        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", " ".join(quote)))
            continue

        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                # Skip the |---|---| separator row.
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append(("table", rows))
            continue

        if line.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].strip().startswith("- ")
                                      or (items and lines[i].startswith("  ")
                                          and lines[i].strip())):
                s = lines[i].strip()
                if s.startswith("- "):
                    items.append(s[2:])
                else:
                    items[-1] += " " + s
                i += 1
            blocks.append(("bullets", items))
            continue

        # Paragraph: consume until a blank line or the start of another block.
        para = []
        while i < len(lines):
            s = lines[i].strip()
            if (not s or s.startswith(("#", "|", "- ", "![", ">"))) or s == "---":
                break
            para.append(s)
            i += 1
        blocks.append(("para", " ".join(para)))
    return blocks


def build(md_path: Path, out_path: Path) -> int:
    blocks = parse_blocks(md_path.read_text())
    with PdfPages(out_path) as pdf:
        doc = Doc(pdf)
        for kind, payload in blocks:
            if kind == "pagebreak":
                doc.break_page()
            elif kind == "heading":
                render_heading(doc, *payload)
            elif kind == "para":
                render_paragraph(doc, payload)
            elif kind == "quote":
                render_quote(doc, payload)
            elif kind == "bullets":
                render_bullets(doc, payload)
            elif kind == "table":
                render_table(doc, payload)
            elif kind == "figure":
                _FIG_COUNT["n"] += 1
                render_figure(doc, payload[0], payload[1])
        doc.page.close()
        pages = doc.n

        info = pdf.infodict()
        info["Title"] = "Transferring a Dex3-1 manipulation policy to BrainCo Revo2 Touch hands"
        info["Subject"] = "Cross-embodiment policy transfer and GR00T N1.7 fine-tuning"
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", default=str(HERE / "assignment1_report.md"))
    ap.add_argument("--out", default=str(HERE / "assignment1_report.pdf"))
    args = ap.parse_args()

    md_path, out_path = Path(args.md), Path(args.out)
    if not md_path.exists():
        raise SystemExit(f"missing {md_path}")

    pages = build(md_path, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} -- {pages} pages, {_FIG_COUNT['n']} figures, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
