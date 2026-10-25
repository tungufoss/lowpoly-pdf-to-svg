"""Export: the last step, turning the finished review SVG into a cut-ready file.

    python -m src.lowpoly_pdf_to_svg.export --config data/bowser/bowser.yaml

reads data/bowser/bowser.svg and edge-numbers.csv (both as left by make, make fix and
the piece review) and writes

* data/bowser/bowser-export.svg: the Pieces layer only (Source and Review dropped),
  every piece filled with its cardstock colour, and every edge number replaced by a
  short label such as 2A. Two edges that glue together share a label, and no other
  edge has it, so the PDF's three-digit numbers become two characters for the
  Cricut pen. Layers and the per-piece Cut, Score and Draw groups stay as they are.
* data/bowser/bowser-labels.csv: which PDF number and which edges each label stands
  for, so a label traces back to the template.

The review SVG, the check page and the fix loop keep the PDF numbers; only this file
is renumbered. Labels are handed out in PDF-number order from an alphabet without
look-alikes (0/O, 1/I, Q; G reads as 6 in pen; 9 is an upside-down 6), all of one width and
each with at least one letter so none reads as a PDF number: two characters cover 851
glue pairs, three beyond that. The export follows the contract in
report/export-contract.qmd; `--check FILE` tests a file against it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from itertools import islice, product
from pathlib import Path

import yaml

SVG, INKSCAPE = "http://www.w3.org/2000/svg", "http://www.inkscape.org/namespaces/inkscape"
for prefix, uri in (("", SVG), ("xlink", "http://www.w3.org/1999/xlink"), ("inkscape", INKSCAPE),
                    ("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")):
    ET.register_namespace(prefix, uri)
LABEL = f"{{{INKSCAPE}}}label"

# Cardstock colours used when the YAML material has no `color:`.
DEFAULT_COLORS = {"red": "#c62828", "green": "#2e7d32", "yellow": "#f9d71c", "black": "#212121",
                  "white": "#ffffff", "beige": "#e8d9b5", "blue": "#1565c0", "orange": "#ef6c00",
                  "brown": "#6d4c41", "grey": "#9e9e9e", "gray": "#9e9e9e", "pink": "#f48fb1",
                  "purple": "#6a1b9a", "rust": "#b7410e", "gold": "#d4a017"}
FALLBACK_COLOR = "#cccccc"
CONTRACT = "lowpoly-export/1"  # bump when the attributes below change meaning


class LabelAlphabet:
    """Short labels of one width, at least two characters (2A, not A next to 2A), as few
    characters as the number of glue pairs allows."""

    # No 0/O, 1/I or Q (look alike), no G (reads as 6 in pen; Helga picked 6), no 9 (an upside-down 6).
    SYMBOLS = "12345678ABCDEFHJKLMNPRSTUVWXYZ"

    def __init__(self, symbols: str = SYMBOLS, shortest: int = 2):
        self.symbols, self.shortest = symbols, shortest

    def take(self, count: int) -> list[str]:
        width = self.shortest
        letters = sum(not c.isdigit() for c in self.symbols)
        while len(self.symbols) ** width - (len(self.symbols) - letters) ** width < count:
            width += 1
        # Labels need a letter, so none can be mistaken for a PDF number.
        labels = ("".join(chars) for chars in product(self.symbols, repeat=width))
        return list(islice((label for label in labels if not label.isdigit()), count))


@dataclass
class Edge:
    """One printed number on one piece, as a row of the edge table."""

    number: str
    piece: str
    x_mm: float  # page coordinates, as in the edge table
    y_mm: float
    partner: str
    status: str
    side: str  # "tab" or "edge"
    line: tuple[float, ...] = ()  # the edge the number marks: ax ay bx by, page mm


@dataclass
class GluePair:
    """The edges that share one label: normally a tab and the edge it glues to."""

    number: str
    edges: list[Edge] = field(default_factory=list)
    label: str = ""
    frozen_label: str = ""

    @property
    def status(self) -> str:
        return self.edges[0].status if self.edges else "no edge row"


class GluePairs:
    """Groups the edge table into glue pairs and gives each pair its short label."""

    def __init__(self, edges: list[Edge]):
        pairs: dict[tuple, GluePair] = {}
        for index, edge in enumerate(edges):
            key = ((edge.number, *sorted((edge.piece, edge.partner))) if edge.partner
                   else (edge.number, edge.piece, index))
            pairs.setdefault(key, GluePair(edge.number)).edges.append(edge)
        # Labels run consecutively through the sheets: a pair is numbered at the first piece
        # (page, then piece on the page) that carries it, so 1A, 1B, ... start on the first sheet.
        self.pairs = sorted(pairs.values(), key=lambda p: (min(e.piece for e in p.edges), _numeric(p.number)))
        self._orphans: dict[str, GluePair] = {}

    @classmethod
    def read(cls, path: Path) -> "GluePairs":
        with path.open(newline="", encoding="utf-8") as handle:
            return cls([Edge(row["number"], row["piece"], float(row["x_mm"]), float(row["y_mm"]),
                             row["partner_piece"], row["status"], row["side"],
                             tuple(float(v) for v in (row.get("line_mm") or "").split()))
                        for row in csv.DictReader(handle)])

    def find(self, piece: str, number: str, x_mm: float, y_mm: float, used: set[int]) -> tuple[GluePair, Edge | None]:
        """The pair of the nearest unused edge row with this piece and number; a number with
        no row gets a pair of its own, shared by every other such copy of that number."""
        best, distance = None, math.inf
        for pair in self.pairs:
            for edge in pair.edges:
                if edge.piece == piece and edge.number == number and id(edge) not in used:
                    d = math.hypot(edge.x_mm - x_mm, edge.y_mm - y_mm)
                    if d < distance:
                        best, distance = (pair, edge), d
        if best:
            used.add(id(best[1]))
            return best
        return self._orphans.setdefault(number, GluePair(number)), None

    def assign(self, alphabet: LabelAlphabet) -> None:
        everything = self.pairs + sorted(self._orphans.values(), key=lambda p: _numeric(p.number))
        reserved = {pair.frozen_label for pair in everything if pair.frozen_label}
        # Reserve the existing sequence positions so external joins never change when
        # self-joins get letters. Letters are globally unique, as the export contract
        # and downstream consumers identify pairs by label alone.
        self_joins = [pair for pair in everything if not pair.frozen_label and len(pair.edges) == 2
                      and pair.edges[0].piece == pair.edges[1].piece
                      and all(edge.partner == edge.piece for edge in pair.edges)]
        letters = [symbol for symbol in alphabet.symbols if symbol.isalpha() and symbol not in reserved]
        if len(self_joins) > len(letters):
            raise ValueError(f"{len(self_joins)} self-join pairs exceed the {len(letters)} "
                             "available single-letter labels; split the export into smaller models")
        normal = alphabet.take(len(everything))
        used = set(reserved)
        deferred = []
        for pair, label in zip(everything, normal):
            if pair.frozen_label:
                pair.label = pair.frozen_label
            elif label not in used:
                pair.label = label
                used.add(label)
            else:
                deferred.append(pair)
        available = iter(label for label in alphabet.take(len(everything) + len(reserved)) if label not in used)
        for pair in deferred:
            pair.label = next(available)
        for pair, label in zip(self_joins, letters):
            pair.label = label

    def freeze(self, records: list[dict], used: set[int]) -> None:
        """Pin complete pair identities from already-cut edges; ambiguity is fatal."""
        grouped = {}
        for row in records:
            key = (row["piece_id"], row["pdf_number"], row["edge"])
            grouped.setdefault(key, []).append(row["label"])
        owners = {}
        for key, labels in grouped.items():
            matches = [(pair, edge) for pair in self.pairs for edge in pair.edges
                       if (edge.piece, edge.number, edge.side) == key and id(edge) in used]
            if len(matches) != len(labels) or len(set(labels)) != 1:
                raise ValueError(f"stale or ambiguous frozen labels for {key}")
            label = labels[0]
            if not label or any(c not in LabelAlphabet.SYMBOLS for c in label):
                raise ValueError(f"unsupported frozen label {label!r}")
            for pair, _ in matches:
                if pair.frozen_label and pair.frozen_label != label:
                    raise ValueError(f"conflicting frozen labels for {key}")
                if label in owners and owners[label] is not pair:
                    raise ValueError(f"frozen label {label} identifies multiple glue pairs")
                owners[label] = pair
                pair.frozen_label = label

    def write(self, path: Path) -> None:
        everything = self.pairs + list(self._orphans.values())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "pdf_number", "status", "pieces"])
            for pair in sorted(everything, key=lambda p: (len(p.label), p.label)):
                writer.writerow([pair.label, pair.number, pair.status, " ".join(e.piece for e in pair.edges)])


class Materials:
    """Per material id from the YAML: name and fill colour (the `color:` if given, else a
    named default), plus the source PDF for provenance."""

    def __init__(self, config: Path):
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        frozen = raw.get("export", {}).get("frozen_labels")
        self.frozen_labels = (json.loads((config.parent / frozen).read_text(encoding="utf-8"))["labels"]
                              if frozen else [])
        self.project = str(raw["project"])
        self.pdf = str(raw["source"]["pdf"])
        self.names = {str(m["id"]): str(m["name"]) for m in raw["materials"]}
        self.colors = {str(m["id"]): str(m.get("color") or DEFAULT_COLORS.get(str(m["id"]).lower(), FALLBACK_COLOR))
                       for m in raw["materials"]}

    def color(self, material: str) -> str:
        return self.colors.get(material, FALLBACK_COLOR)

    def name(self, material: str) -> str:
        return self.names.get(material, material)


class ExportWriter:
    """Rewrites the review SVG into the export SVG, following the export contract
    (report/reference.qmd): every fact a consumer needs is a data-* attribute on the
    piece group or on each path and text, never only a style or the element order."""

    PIECE_ID = re.compile(r"piece-(\d+)-\d+$")

    def __init__(self, pairs: GluePairs, materials: Materials, alphabet: LabelAlphabet = LabelAlphabet(),
                 twins: dict[str, str] | None = None, label_size: float = 5.0, number_size: float = 10.0):
        self.pairs, self.materials, self.alphabet, self.twins = pairs, materials, alphabet, twins or {}
        # One size for every edge label (Helga: same font size); the pen font's cap height is 0.75 of it.
        self.sizes = (label_size,)
        self.number_size = number_size

    def export(self, source: Path, target: Path) -> dict[str, int]:
        tree = ET.parse(source)
        root = tree.getroot()
        for layer in [g for g in root.findall(f"{{{SVG}}}g") if g.get("id") in ("source", "review")]:
            root.remove(layer)
        title = root.find(f"{{{SVG}}}title")
        if title is not None:
            title.text = f"{title.text} (export)"
        root.set("data-contract", CONTRACT)
        root.set("data-units", "mm")
        root.set("data-project", self.materials.project)
        offsets = {page.get("id"): (float(page.get("x")), float(page.get("y")))
                   for page in root.iter(f"{{{INKSCAPE}}}page")}

        texts, used = [], set()
        pieces = [g for g in root.iter(f"{{{SVG}}}g") if g.get("data-material") is not None]
        numbers = PieceNumbers([g.get("id") for g in pieces], self.twins)
        for piece in pieces:
            match = self.PIECE_ID.match(piece.get("id", ""))
            page = int(match.group(1)) if match else 0
            ox, oy = offsets.get(f"page-{page:02d}", (0.0, 0.0))
            self._describe(piece, page, ox, oy)
            outline, on_edge, taken = _outline(piece), {}, []
            for text in piece.iter(f"{{{SVG}}}text"):
                number = (text.text or "").strip()
                pair, edge = self.pairs.find(piece.get("id"), number,
                                             float(text.get("x")) - ox, float(text.get("y")) - oy, used)
                if edge:
                    text.set("data-edge", edge.side)  # tab or plain edge, from the edge table
                    if len(edge.line) == 4 and outline:
                        on_edge.setdefault(edge.line, []).append(text)
                texts.append((text, pair))
            # Shortest edges first, as they have the least room; two numbers on one edge spread along it.
            for line, group in sorted(on_edge.items(), key=lambda item: math.hypot(item[0][2] - item[0][0],
                                                                                 item[0][3] - item[0][1])):
                group.sort(key=lambda e: (float(e.get("x")) - line[0] - ox) * (line[2] - line[0])
                           + (float(e.get("y")) - line[1] - oy) * (line[3] - line[1]))
                for k, text in enumerate(group, 1):
                    _place(text, line, ox, oy, outline, k / (len(group) + 1), text.get("data-edge") == "tab",
                           self.sizes, taken=taken)
            numbers.add(piece, outline, self.number_size, taken)
        self.pairs.freeze(self.materials.frozen_labels, used)
        self.pairs.assign(self.alphabet)
        for text, pair in texts:
            text.set("data-pdf-number", text.text or "")
            text.set("data-label", pair.label)
            text.set("data-pair-status", pair.status)
            text.text = pair.label
            for attribute in ("fill", "data-review"):  # the review's orange "unsure" mark
                text.attrib.pop(attribute, None)

        ET.indent(tree, space="")
        target.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target, encoding="UTF-8", xml_declaration=True)
        return {"pieces": len(pieces), "labels": len(texts),
                "pairs": len({id(pair) for _, pair in texts}),
                "longest": max((len(pair.label) for _, pair in texts), default=0)}

    def _describe(self, piece: ET.Element, page: int, ox: float, oy: float) -> None:
        """Provenance and material on the piece group; operation on every path and text.
        The cardstock fills the piece: outline and holes become one even-odd path, so
        holes stay open, and the numbers turn white on a dark card to stay readable."""
        material = piece.get("data-material")
        color = self.materials.color(material)
        piece.set(LABEL, piece.get(LABEL, "").split(" ⚠")[0])
        piece.set("data-piece", piece.get("id"))
        piece.set("data-material-name", self.materials.name(material))
        piece.set("data-color", color)
        piece.set("data-source-pdf", self.materials.pdf)
        piece.set("data-source-page", str(page))
        piece.set("data-page-origin-mm", f"{ox:.3f} {oy:.3f}")
        for group in piece.iter(f"{{{SVG}}}g"):
            operation = group.get("data-operation")
            if operation == "cut":
                paths = group.findall(f"{{{SVG}}}path")
                if paths:
                    paths[0].set("d", " ".join(p.get("d") for p in paths))
                    for extra in paths[1:]:
                        group.remove(extra)
                group.set("style", _style(group.get("style", ""), fill=color, **{"fill-rule": "evenodd"}))
                for path in group.findall(f"{{{SVG}}}path"):
                    path.set("data-operation", "cut")
            elif operation == "score":
                for fold in group.iter(f"{{{SVG}}}g"):
                    for path in fold.findall(f"{{{SVG}}}path"):
                        path.set("data-operation", "score")
                        path.set("data-fold", fold.get("data-fold", ""))
                        path.attrib.pop("data-review", None)
            elif operation == "draw":
                for text in group.findall(f"{{{SVG}}}text"):
                    text.set("data-operation", "draw")
                if _luminance(color) < 0.35:
                    group.set("style", _style(group.get("style", ""), fill="#ffffff"))


def _outline(piece: ET.Element) -> list[tuple[float, float]]:
    """Outer cut outline of a piece (first subpath of its cut path), document mm."""
    for path in piece.iter(f"{{{SVG}}}path"):
        if path.get("data-operation") == "cut":
            first = re.split(r"[Mm]", path.get("d", ""))[1] if "M" in path.get("d", "") else ""
            v = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", first)]
            return list(zip(v[::2], v[1::2]))
    return []


def _inside(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y, inside = p[0], p[1], False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _depth(mx: float, my: float, nx: float, ny: float, poly: list[tuple[float, float]]) -> float:
    """How far the piece reaches from (mx, my) along (nx, ny): 0 if that side is outside."""
    if not _inside((mx + nx * 0.05, my + ny * 0.05), poly):
        return 0.0
    best = math.inf
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = x2 - x1, y2 - y1
        den = nx * ey - ny * ex
        if abs(den) < 1e-12:
            continue
        t = ((x1 - mx) * ey - (y1 - my) * ex) / den
        s = ((x1 - mx) * ny - (y1 - my) * nx) / den
        if t > 0.05 and 0 <= s <= 1:
            best = min(best, t)
    return 0.0 if best == math.inf else best


def _place(text: ET.Element, line: tuple[float, ...], ox: float, oy: float,
           outline: list[tuple[float, float]], at: float = 0.5, tab: bool = False,
           sizes: tuple[float, ...] = (2.0,), chars: int = 2,
           taken: list | None = None) -> None:
    """Stand the label on the edge it marks: centred at `at` along the edge, baseline along the
    edge, its top pointing away from the edge (perpendicular to it), inside the cut outline.
    A cut edge has the piece on one side only; a tab's number marks its fold, and the label goes
    on the tab (the shallower side), centred across it when it fits, else on the piece side. All
    labels have one size (`sizes[0]`); never where the PDF printed the number. If the centre is
    taken it slides along the edge, and as a last resort it may stick out past the outline rather
    than cover another label. `taken` holds the boxes of labels already placed on the
    piece; a label avoids them and adds its own."""
    taken = [] if taken is None else taken
    ax, ay, bx, by = line[0] + ox, line[1] + oy, line[2] + ox, line[3] + oy
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-6:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    mx, my = ax + (bx - ax) * at, ay + (by - ay) * at
    nx, ny = -uy, ux
    plus, minus = _depth(mx, my, nx, ny, outline), _depth(mx, my, -nx, -ny, outline)
    if plus and minus:  # both sides inside: a fold; the tab is the shallower side
        flip = (minus < plus) if tab else (minus > plus)
    elif plus or minus:
        flip = not plus
    else:  # the line is off the outline: keep the side the PDF printed it on
        px, py = float(text.get("x")), float(text.get("y"))
        flip = (px - mx) * nx + (py - my) * ny < 0
    other = minus if plus else 0.0  # depth on the far side, kept for a fold label that has no room
    if flip:
        nx, ny, plus, other = -nx, -ny, minus, plus
    sides = [(nx, ny, plus)] + ([(-nx, -ny, other)] if plus and other else [])

    def fits(size: float, at_: float, side, inside: bool = True):
        """Centre of a label of this size at `at_` along the edge, on this side, that fits, or None."""
        sx_, sy_, depth = side
        on_tab = tab and depth and side is sides[0]
        # On a narrow tab the box is the pen's cap height (0.75 of the size) with 0.2 mm to spare,
        # so the label still fits between the fold and the tab's edge (Helga: 4L below its line).
        half_w, half_h = 0.6 * size * chars / 2, (0.375 if on_tab else 0.4) * size
        px, py = ax + (bx - ax) * at_, ay + (by - ay) * at_
        start = max(half_h + 0.2, depth / 2) if on_tab else half_h + 0.5
        for step in range(16):
            d = start + 0.25 * step if not on_tab or step < 8 else half_h + 0.2 + 0.25 * (step - 8)
            cx, cy = px + sx_ * d, py + sy_ * d
            corners = [(cx + kx * ux * half_w + ky * sx_ * half_h, cy + kx * uy * half_w + ky * sy_ * half_h)
                       for kx in (-1, 1) for ky in (-1, 1)]
            if (not inside or all(_inside(c, outline) for c in corners)) \
                    and not any(_overlap(corners, box) for box in taken):
                return cx, cy
        return None

    # Sized for the short label (chars), not the PDF number the text still holds. First on its own
    # side (a tab label on the tab), centred, then slid along the edge; only then the other side of
    # a fold (Helga: a tab label may sit on the piece); last, allowed past the outline, but never on
    # top of another label.
    slid = [a for k in (1, 2, 3, 4) for a in (at - 0.1 * k, at + 0.1 * k) if 0.08 < a < 0.92]
    size = sizes[0]
    attempts = ([(a, side, True) for side in sides for a in [at] + slid]
                + [(a, side, False) for side in sides for a in [at] + slid])
    placed, chosen = None, sides[0]
    for a_, side, inside in attempts:
        spot = fits(size, a_, side, inside)
        if spot:
            placed, chosen, (nx, ny, _) = (size, *spot), side, side
            break
    if not placed:  # nowhere free: centred just off the edge
        placed = (size, mx + nx * (0.4 * size + 0.5), my + ny * (0.4 * size + 0.5))
    size, cx, cy = placed
    half_w, half_h = 0.6 * size * chars / 2 + 1.0, 0.4 * size + 0.3  # with air round it, more along the line so 4S 5F never read as one
    taken.append([(cx + sx * ux * half_w + sy * nx * half_h, cy + sx * uy * half_w + sy * ny * half_h)
                  for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    angle = math.degrees(math.atan2(nx, -ny))  # rotate(angle) turns the text's up (0,-1) onto (nx, ny)
    if tab and len(sides) == 2 and chosen is sides[0]:  # on the tab: reads from the tab's outer edge,
        angle = (angle + 180) % 360                    # its top towards the fold (Helga, 5E)
    text.set("x", f"{cx:.3f}")
    text.set("y", f"{cy:.3f}")
    text.set("font-size", f"{size:.2f}")
    text.set("text-anchor", "middle")
    text.set("transform", f"rotate({angle:.1f} {cx:.3f} {cy:.3f})")


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """Do two convex quadrilaterals overlap (separating-axis test; corner order is irrelevant)."""
    for poly in (a, b):
        for i in range(len(poly)):
            for j in range(i + 1, len(poly)):
                ex, ey = poly[j][0] - poly[i][0], poly[j][1] - poly[i][1]
                pa = [(q[0] * -ey + q[1] * ex) for q in a]
                pb = [(q[0] * -ey + q[1] * ex) for q in b]
                if max(pa) < min(pb) or max(pb) < min(pa):
                    return False
    return True


def _clearance(p: tuple[float, float], segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / (dx * dx + dy * dy or 1)))
        best = min(best, math.hypot(p[0] - x1 - t * dx, p[1] - y1 - t * dy))
    return best


class PieceNumbers:
    """A number in the middle of every piece, for finding pieces on the cut sheet. Pieces are
    numbered 1, 2, ... in piece order (page, then piece); a mirror twin (mirror-mismatches.csv)
    shares its twin's number. The first of the two gets a circle round its number, the second a
    square, so 4 in a circle and 4 in a square are the left and right copy of one part."""

    def __init__(self, pieces: list[str], twins: dict[str, str] | None = None):
        twins = twins or {}
        self.numbers, self.frames, count = {}, {}, 0
        for piece in sorted(pieces):
            if piece in self.numbers:
                continue
            count += 1
            self.numbers[piece] = count
            twin = twins.get(piece)
            if twin and twin in pieces and twin not in self.numbers:
                self.numbers[twin] = count
                self.frames[piece], self.frames[twin] = "circle", "square"

    @staticmethod
    def read_twins(path: Path) -> dict[str, str]:
        """piece -> twin, both ways, from mirror-mismatches.csv (absent: no twins)."""
        twins = {}
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    twins.setdefault(row["piece"], row["twin"])
                    twins.setdefault(row["twin"], row["piece"])
        return twins

    def add(self, piece: ET.Element, outline: list[tuple[float, float]], biggest: float = 10.0,
            labels: list | None = None) -> None:
        """Write the number (and its frame) into the piece's Draw group, at the point of the piece
        farthest from its outline and from the edge-label boxes (`labels`), as big as that room
        allows. Fold lines may run through it: every label has to stay readable, the folds do not."""
        pid, draw = piece.get("id"), piece.find(f".//{{{SVG}}}g[@data-operation='draw']")
        if pid not in self.numbers or draw is None or len(outline) < 3:
            return
        text, frame = str(self.numbers[pid]), self.frames.get(pid)
        segments = list(zip(outline, outline[1:] + outline[:1]))
        for box in labels or []:
            segments += list(zip(box, box[1:] + box[:1]))
        best, room = _widest_spot(outline, segments)
        if not best:
            return
        # Half the number's width in font sizes (pen font: 0.69 per character, cap 0.75), plus the frame.
        half = max(0.69 * len(text) - 0.19, 0.75) / 2 * (1.45 if frame else 1.0)
        size = max(3.0, min(biggest, (room - 0.5) / half))
        x, y = best
        colour = _style_value(draw.get("style", ""), "fill") or "#000000"
        element = ET.SubElement(draw, f"{{{SVG}}}text", {"x": f"{x:.3f}", "y": f"{y:.3f}", "font-size": f"{size:.2f}",
                                                       "dominant-baseline": "central", "text-anchor": "middle",
                                                       "data-operation": "draw", "data-label": text,
                                                       "data-piece-number": text})
        element.text = text
        if set(text) <= set("01689") and set(text) & set("69"):  # 6 vs 9, 16 vs 91: underline
            w, uy = (0.69 * len(text) - 0.19) / 2 * size, y + 0.5 * size
            ET.SubElement(draw, f"{{{SVG}}}path", {"d": f"M {x - w:.3f},{uy:.3f} {x + w:.3f},{uy:.3f}",
                                                 "style": f"fill:none;stroke:{colour};stroke-width:0.3",
                                                 "data-operation": "draw", "data-mark": "underline",
                                                 "data-piece-number": text})
        if frame:
            r = half * size * (1.15 if frame == "circle" else 1.0)
            ring = ([(x + r * math.cos(k * math.pi / 12), y + r * math.sin(k * math.pi / 12)) for k in range(25)]
                    if frame == "circle" else [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r), (x - r, y - r)])
            ET.SubElement(draw, f"{{{SVG}}}path", {"d": "M " + " ".join(f"{px:.3f},{py:.3f}" for px, py in ring),
                                                 "style": f"fill:none;stroke:{colour};stroke-width:0.3",
                                                 "data-operation": "draw", "data-mark": frame,
                                                 "data-piece-number": text})


def _widest_spot(outline: list[tuple[float, float]], segments) -> tuple[tuple[float, float] | None, float]:
    """The inside point farthest from every segment: a coarse grid, then a finer one around the best."""
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    best, room, step = None, 0.0, max(x1 - x0, y1 - y0) / 24
    cells = [(x0 + (i + 0.5) * step, y0 + (j + 0.5) * step)
             for i in range(int((x1 - x0) / step) + 1) for j in range(int((y1 - y0) / step) + 1)]
    for _ in range(3):
        for p in cells:
            if _inside(p, outline):
                c = _clearance(p, segments)
                if c > room:
                    best, room = p, c
        if not best:
            break
        step /= 4
        cells = [(best[0] + i * step, best[1] + j * step) for i in range(-4, 5) for j in range(-4, 5)]
    return best, room


def _style_value(style: str, key: str) -> str:
    return dict(part.split(":", 1) for part in style.split(";") if ":" in part).get(key, "")


class ExportContract:
    """Checks a file against the export contract, as an independent consumer would:
    only data-* attributes, no reliance on styles or element order."""

    def __init__(self, path: Path):
        self.root = ET.parse(path).getroot()

    def errors(self) -> list[str]:
        root, problems = self.root, []
        if root.get("data-contract") != CONTRACT:
            problems.append(f"root data-contract is {root.get('data-contract')!r}, expected {CONTRACT!r}")
        if root.get("data-units") != "mm" or not str(root.get("width", "")).endswith("in"):
            problems.append("root needs data-units=\"mm\" and a width/height in inches")
        for tag in ("clipPath", "mask", "use", "symbol", "image"):
            if root.find(f".//{{{SVG}}}{tag}") is not None:
                problems.append(f"unsupported element <{tag}>")
        seen, labels = set(), {}
        for piece in root.iter(f"{{{SVG}}}g"):
            if piece.get("data-piece") is None:
                if piece.get("transform"):
                    problems.append(f"group {piece.get('id')} has a transform")
                continue
            pid = piece.get("data-piece")
            if pid in seen:
                problems.append(f"{pid}: piece id is not unique")
            seen.add(pid)
            for key in ("data-material", "data-color", "data-source-pdf", "data-source-page", "data-page-origin-mm"):
                if not piece.get(key):
                    problems.append(f"{pid}: missing {key}")
            if piece.get("transform"):
                problems.append(f"{pid}: piece group has a transform")
            leaves = [e for e in piece.iter() if e.tag in (f"{{{SVG}}}path", f"{{{SVG}}}text")]
            cuts = [e for e in leaves if e.get("data-operation") == "cut"]
            if len(cuts) != 1:
                problems.append(f"{pid}: {len(cuts)} cut paths, expected exactly 1")
            for leaf in leaves:
                operation = leaf.get("data-operation")
                if operation not in ("cut", "score", "draw"):
                    problems.append(f"{pid}: {leaf.tag.split('}')[1]} without data-operation")
                if operation == "score" and leaf.get("data-fold") not in ("mountain", "valley"):
                    problems.append(f"{pid}: score path without data-fold mountain/valley")
                if operation == "draw" and leaf.tag == f"{{{SVG}}}path":
                    if leaf.get("data-mark") not in ("circle", "square", "underline") or not leaf.get("data-piece-number"):
                        problems.append(f"{pid}: draw path that is not a piece-number circle, square or underline")
                elif operation == "draw":
                    if not leaf.get("data-label") or leaf.get("data-label") != (leaf.text or ""):
                        problems.append(f"{pid}: label text and data-label differ")
                    if not leaf.get("data-piece-number"):
                        labels.setdefault(leaf.get("data-label"), []).append(leaf.get("data-pair-status"))
                if leaf.get("transform") and not (leaf.tag == f"{{{SVG}}}text"
                                                  and leaf.get("transform").startswith("rotate(")):
                    problems.append(f"{pid}: transform other than a text rotate")
        if not seen:
            problems.append("no piece groups (g with data-piece)")
        for label, statuses in labels.items():
            if len(statuses) > 2:
                problems.append(f"label {label} is on {len(statuses)} edges")
        return problems


def _style(style: str, **changes: str) -> str:
    items = dict(part.split(":", 1) for part in style.split(";") if ":" in part)
    items.update(changes)
    return ";".join(f"{key}:{value}" for key, value in items.items())


def _luminance(color: str) -> float:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return 1.0
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _numeric(number: str) -> tuple[int, str]:
    return (int(number), number) if number.isdigit() else (sys.maxsize, number)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the cut-ready export SVG with short edge labels and cardstock fills.")
    parser.add_argument("--config", type=Path, default=Path("data/bowser/bowser.yaml"), help="project YAML manifest")
    parser.add_argument("--output", type=Path, help="model folder (default: the YAML's folder)")
    parser.add_argument("--svg", type=Path, help="review SVG to export (default: <folder>/<project>.svg)")
    parser.add_argument("--check", type=Path, metavar="EXPORT", help="only check an export SVG against the contract")
    parser.add_argument("--label-size", type=float, default=5.0, help="edge-label font size, mm (default 5)")
    parser.add_argument("--number-size", type=float, default=10.0, help="largest piece-number font size, mm (default 10)")
    args = parser.parse_args()
    if args.check:
        problems = ExportContract(args.check).errors()
        print("\n".join(problems) or f"{args.check}: follows {CONTRACT}")
        return 1 if problems else 0
    folder = args.output or args.config.parent
    project = args.config.stem
    source, edges = args.svg or folder / f"{project}.svg", folder / "edge-numbers.csv"
    for path in (source, edges):
        if not path.is_file():
            parser.error(f"{path} not found; run make first")
    pairs = GluePairs.read(edges)
    target = folder / f"{project}-export.svg"
    twins = PieceNumbers.read_twins(folder / "mirror-mismatches.csv")
    counts = ExportWriter(pairs, Materials(args.config), twins=twins, label_size=args.label_size,
                          number_size=args.number_size).export(source, target)
    pairs.write(folder / f"{project}-labels.csv")
    problems = ExportContract(target).errors()
    print(f"{target}: {counts['pieces']} pieces filled, {counts['labels']} numbers -> "
          f"{counts['pairs']} labels of at most {counts['longest']} characters; "
          f"key in {folder / f'{project}-labels.csv'}")
    for problem in problems:
        print(f"contract: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
