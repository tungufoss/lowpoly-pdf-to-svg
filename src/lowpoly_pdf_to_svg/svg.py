"""Stage 3: write the whole project as one Inkscape review SVG.

Every PDF page becomes an Inkscape page (Inkscape 1.2+), one row per cardstock.
The document size is given in inches, as Cricut Design Space expects; drawing
coordinates (the viewBox) stay in millimetres, so both read the true size.
Layers span the whole file:

* "Source" layer (locked, faint): the rendered pages, for checking traces by eye.
* "Pieces" layer with one sublayer per cardstock: one group per piece, each holding
  its Cut, Score and Draw groups, so a piece moves as a unit and keeps its
  cardstock in data-material.
* "Review" layer (hidden): artwork outside every piece, such as credits or an
  outline the detector could not close. Drag anything that belongs to a piece back.

Folds are drawn dash-dot (mountain) or dashed (valley) so they read like the
template in Inkscape; Design Space ignores dash styles and needs Score assigned.
Delete the Source and Review layers before uploading to Design Space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from .config import Material
from .model import FoldKind, Label, PageLayout, Piece, Point
from .render import PageImage

CUT_STYLE = "fill:none;stroke:#000000;stroke-width:0.25"
FOLD_STYLE = {
    FoldKind.MOUNTAIN: "fill:none;stroke:#1f5fbf;stroke-width:0.25;stroke-dasharray:2.5,0.8,0.4,0.8",
    FoldKind.VALLEY: "fill:none;stroke:#2e8b57;stroke-width:0.25;stroke-dasharray:1.5,1",
}
DRAW_STYLE = "fill:#8e24aa;stroke:none;font-family:Arial,sans-serif;font-weight:bold;text-anchor:middle"
UNSURE_FILL = "#e65100"  # numbers the reader was unsure of
REVIEW_STYLE = "fill:none;stroke:#e65100;stroke-width:0.2"
PAGE_GAP_MM = 20.0
MM_PER_INCH = 25.4


@dataclass
class PageResult:
    image: PageImage
    layout: PageLayout
    material: Material


class InkscapeSvgWriter:
    def __init__(self, title: str, materials: dict[str, Material] | None = None):
        self.title = title
        self.pages: list[PageResult] = []
        self.materials = materials or {}   # material id -> Material, for pieces moved off their page's cardstock

    def add(self, image: PageImage, layout: PageLayout, material: Material) -> None:
        self.pages.append(PageResult(image, layout, material))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(path.parent), encoding="utf-8")

    def render(self, directory: Path) -> str:
        # One row of pages per cardstock, rows in the order the materials first appear.
        rows: dict[str, list[PageResult]] = {}
        for page in self.pages:
            rows.setdefault(page.material.id, []).append(page)
        offsets: dict[int, tuple[float, float]] = {}
        width, y = 1.0, 0.0
        for row in rows.values():
            x = 0.0
            for page in row:
                offsets[id(page)] = (x, y)
                x += page.image.size_mm[0] + PAGE_GAP_MM
            width = max(width, x - PAGE_GAP_MM)
            y += max(p.image.size_mm[1] for p in row) + PAGE_GAP_MM
        height = max(y - PAGE_GAP_MM, 1.0)
        # The template prints every number at one size; OCR boxes vary with rotation and
        # fit, so all labels take the median height instead of their own box.
        heights = sorted(label.height * page.image.mm_per_px for page in self.pages
                         for piece in page.layout.pieces for label in piece.labels)
        label_mm = heights[len(heights) // 2] if heights else 1.5

        inkscape_pages, sources, reviews = [], [], []
        by_material: dict[str, tuple[Material, list[str]]] = {}
        for page in self.pages:
            ox, oy = offsets[id(page)]
            w, h = page.image.size_mm
            number = page.layout.page
            inkscape_pages.append(f'<inkscape:page id="page-{number:02d}" inkscape:label="Page {number:02d} · '
                                  f'{escape(page.material.name)}" x="{ox:.3f}" y="{oy:.3f}" width="{w:.3f}" height="{h:.3f}"/>')
            href = Path(os.path.relpath(page.image.path, directory)).as_posix()
            sources.append(f'<image id="source-{number:02d}" xlink:href={quoteattr(href)} x="{ox:.3f}" y="{oy:.3f}" '
                           f'width="{w:.3f}" height="{h:.3f}" preserveAspectRatio="none"/>')
            pen = _Pen(page.image.mm_per_px, ox, oy, label_mm)
            for piece in page.layout.pieces:
                # A piece the YAML moves to another cardstock goes in that cardstock's layer.
                material = self.materials.get(piece.material, page.material)
                by_material.setdefault(material.id, (material, []))[1].append(self._piece(piece, pen))
            stray = [pen.path(points) for points in page.layout.stray_lines]
            stray += [pen.box(label.box) for label in page.layout.stray_labels]
            reviews.append(f'<g id="review-{number:02d}" inkscape:label="Page {number:02d}">\n' + "\n".join(stray) + "\n</g>")

        materials = "\n".join(
            f'<g id="material-{escape(m.id)}" inkscape:groupmode="layer" inkscape:label="{escape(m.name)}">\n'
            + "\n".join(pieces) + "\n</g>"
            for m, pieces in by_material.values()
        )
        nl = "\n"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
     width="{width / MM_PER_INCH:.4f}in" height="{height / MM_PER_INCH:.4f}in" viewBox="0 0 {width:.3f} {height:.3f}">
<title>{escape(self.title)}</title>
<sodipodi:namedview inkscape:document-units="mm">
{nl.join(inkscape_pages)}
</sodipodi:namedview>
<g id="source" inkscape:groupmode="layer" inkscape:label="Source" sodipodi:insensitive="true" opacity="0.35">
{nl.join(sources)}
</g>
<g id="pieces" inkscape:groupmode="layer" inkscape:label="Pieces">
{materials}
</g>
<g id="review" inkscape:groupmode="layer" inkscape:label="Review" style="display:none">
<g style="{REVIEW_STYLE}">
{nl.join(reviews)}
</g>
</g>
</svg>
"""

    @staticmethod
    def _piece(piece: Piece, pen: "_Pen") -> str:
        checks = [what for what, needed in (("folds", any(not f.snapped for f in piece.folds)),
                                              ("numbers", any(not l.trusted for l in piece.labels))) if needed]
        flag = f" ⚠ check {' and '.join(checks)}" if checks else ""
        cut = "\n".join(pen.path(points, closed=True) for points in [piece.outline, *piece.holes])
        draw = "\n".join(pen.text(label) for label in piece.labels)
        score = []
        for kind in FoldKind:
            lines = "\n".join(pen.path([f.a, f.b], extra="" if f.snapped else ' data-review="unsnapped"')
                              for f in piece.folds if f.kind is kind)
            score.append(f'<g id="{piece.id}-{kind.value}" inkscape:label="{kind.value.title()} folds" '
                         f'data-fold="{kind.value}" style="{FOLD_STYLE[kind]}">\n{lines}\n</g>')
        return (
            f'<g id="{piece.id}" inkscape:label="{piece.id}{flag}" data-material="{escape(piece.material)}">\n'
            f'<g id="{piece.id}-cut" inkscape:label="Cut" data-operation="cut" style="{CUT_STYLE}">\n{cut}\n</g>\n'
            f'<g id="{piece.id}-score" inkscape:label="Score" data-operation="score">\n' + "\n".join(score) + "\n</g>\n"
            f'<g id="{piece.id}-draw" inkscape:label="Draw" data-operation="draw" style="{DRAW_STYLE}">\n{draw}\n</g>\n'
            f"</g>"
        )


class _Pen:
    """Converts page pixels to document millimetres, shifted onto the page's slot."""

    def __init__(self, mm_per_px: float, offset_mm: float, top_mm: float = 0.0, label_mm: float = 1.5):
        self.scale = mm_per_px
        self.label_mm = label_mm
        self.offset = offset_mm
        self.top = top_mm

    def box(self, box: tuple[int, int, int, int]) -> str:
        x, y, w, h = box
        return self.path([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True)

    def text(self, label: Label) -> str:
        s = self.scale
        cx, cy = label.center[0] * s + self.offset, label.center[1] * s + self.top
        size = self.label_mm / 0.72  # digit height is about 0.72 em
        unsure = "" if label.trusted else f' fill="{UNSURE_FILL}" data-review="ocr"'
        return (f'<text x="{cx:.3f}" y="{cy:.3f}" font-size="{size:.2f}" dominant-baseline="central" '
                f'transform="rotate({label.angle:.1f} {cx:.3f} {cy:.3f})"{unsure}>{escape(label.text)}</text>')

    def path(self, points: list[Point], closed: bool = False, extra: str = "") -> str:
        s, ox, oy = self.scale, self.offset, self.top
        commands = " ".join(f"{x * s + ox:.3f},{y * s + oy:.3f}" for x, y in points)
        return f'<path d="M {commands}{" Z" if closed else ""}"{extra}/>'
