"""Export stage: replace the PDF's traced glue tabs with generated ones.

The template's tabs are narrow and all sit where Pepakura put them. At export every glue
pair of the edge table that has one tab and one plain edge gets a new tab, built from the
finished review SVG:

* wider: `tabs: depth_mm` in the project YAML (default 7.5 mm, half of 15 mm double-sided tape;
  the PDF's are 5.6 mm), with
  sloped ends; where two big pieces meet (`flap_area_mm2`, `flap_edge_mm`) and at stress
  points (the `stress_top` share of seams carrying the most load in fold3d.json, see stress.py,
  plus any listed under `flaps:` as piece ids or PDF numbers), the longest glue of the
  seam gets a flap: the tab plus a paper doll's tongue in the middle of the edge, `flap_mm`
  wide (default 15 mm, one full tape width) reaching `flap_length_mm` (default 18 mm);
* never over a fold: the tab is laid onto the partner's face it glues to and clipped to
  that face, `margin_mm` inside its other edges, so it folds flat onto one face;
* mirrored: a twin's tab sits on the mirror edge of its twin's tab (mirror-mismatches.csv
  pairs the numbers), so left and right copies are cut alike;
* interwoven where a piece glues to its own twin (the centre seam): there the tabs
  alternate between the two pieces along the seam.

Materials listed under `tabs: keep:` (cardstock already cut) keep their traced tabs, so a
recut matches the pieces in hand. Everywhere else a tab stays on the side the PDF chose. Pairs that are not one tab and one
edge, or whose faces cannot be found, keep the traced tab. The review SVG, check page and
3D fold keep the PDF's tabs; only the export changes.
"""

from __future__ import annotations

import csv
import math
import re
from fnmatch import fnmatch
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from .stress import SeamLoad

SVG = "http://www.w3.org/2000/svg"
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
ON_LINE_MM = 1.0      # a fold or corner this close to an edge-table line is that line


@dataclass
class TabStyle:
    depth_mm: float = 7.5      # how far a tab reaches from its fold: half a 15 mm tape (the PDF's are 5.6 mm)
    flap_mm: float = 15.0      # width of a flap's tongue along the edge, one 15 mm tape width (0: none)
    flap_length_mm: float = 18.0    # how far the tongue reaches from the fold (15-20 mm)
    flap_area_mm2: float = 5000.0   # a seam between two pieces this big gets one flap on its longest glue
    flap_edge_mm: float = 25.0      # ... when that glue is at least this long
    stress_top: float = 0.1         # share of seams, by load in the 3D model (fold3d.json), that get a flap
    angle_deg: float = 45.0    # slope of the tab's ends
    margin_mm: float = 0.8     # clearance from the partner face's other edges (folds and cuts)
    gap_mm: float = 1.0        # clearance between two tabs of one piece
    min_mm2: float = 4.0       # a generated tab smaller than this keeps the traced one
    keep: list[str] = field(default_factory=list)   # materials already cut: their traced tabs stay
    flaps: list[str] = field(default_factory=list)  # stress points: piece ids (their seams) or PDF numbers
    extra_flaps: list[str] = field(default_factory=list)  # pieces (ids or patterns) whose plain glue edges
                                                          # get a flap too, beside the partner's tab
    generate: bool = True      # false: keep the traced tabs, add only the extra flaps

    @classmethod
    def from_config(cls, raw: dict | None) -> "TabStyle | None":
        """`tabs:` in the YAML; `tabs: false` (or `generate: false`) keeps the PDF tabs, and
        `generate: false` with `extra_flaps:` keeps them but adds the extra flaps."""
        if raw is False or (isinstance(raw, dict) and raw.get("generate") is False and not raw.get("extra_flaps")):
            return None
        raw = raw if isinstance(raw, dict) else {}
        lists = ("keep", "flaps", "extra_flaps")
        style = cls(**{k: float(v) for k, v in raw.items()
                       if k in cls.__dataclass_fields__ and k not in lists + ("generate",)})
        style.keep, style.flaps, style.extra_flaps = ([str(m) for m in raw.get(k) or []] for k in lists)
        style.generate = raw.get("generate", True) is not False
        return style


@dataclass
class PieceShape:
    """One piece of the review SVG: cut outline, holes and folds, document mm."""
    element: ET.Element
    outline: Polygon
    holes: list[list[tuple[float, float]]]
    folds: list[tuple[LineString, str, ET.Element]]   # line, mountain/valley, its path element
    origin: tuple[float, float]
    body: Polygon | None = None
    faces: list[Polygon] = field(default_factory=list)
    old_tabs: list[Polygon] = field(default_factory=list)

    def line(self, row) -> LineString:
        ax, ay, bx, by = row.line
        ox, oy = self.origin
        return LineString([(ax + ox, ay + oy), (bx + ox, by + oy)])


@dataclass
class Glue:
    """One glue pair: the PDF's tab row and plain-edge row."""
    number: str
    tab: object      # export.Edge rows
    edge: object
    kind: str = ""   # fold style of the PDF tab
    side: str = ""   # piece that gets the new tab
    mirror: "Glue | None" = None
    shape: Polygon | None = None

    @property
    def pieces(self) -> tuple[str, str]:
        return self.tab.piece, self.edge.piece


class Retabber:
    def __init__(self, style: TabStyle, twins_csv: Path | None = None, fold3d: Path | None = None):
        self.style = style
        self.stressed: set[frozenset] = set()
        if fold3d and fold3d.is_file() and style.stress_top > 0:
            self.stressed = SeamLoad(fold3d).top(style.stress_top)
        self.mirror_numbers: dict[tuple[str, str], tuple[str, str]] = {}
        self.twin: dict[str, str] = {}
        if twins_csv and twins_csv.is_file():
            with twins_csv.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.twin.setdefault(row["piece"], row["twin"])
                    self.twin.setdefault(row["twin"], row["piece"])
                    if row.get("twin_number"):
                        self.mirror_numbers[(row["piece"], row["number"])] = (row["twin"], row["twin_number"])
                        self.mirror_numbers[(row["twin"], row["twin_number"])] = (row["piece"], row["number"])
        self.stats: dict[str, int] = defaultdict(int)

    # ---- reading -------------------------------------------------------------------------
    def _shapes(self, root: ET.Element) -> dict[str, PieceShape]:
        offsets = {page.get("id"): (float(page.get("x")), float(page.get("y")))
                   for page in root.iter("{http://www.inkscape.org/namespaces/inkscape}page")}
        shapes = {}
        for g in root.iter(f"{{{SVG}}}g"):
            if g.get("data-material") is None:
                continue
            match = re.match(r"piece-(\d+)-\d+$", g.get("id", ""))
            origin = offsets.get(f"page-{int(match.group(1)):02d}", (0.0, 0.0)) if match else (0.0, 0.0)
            rings, folds = [], []
            for group in g.iter(f"{{{SVG}}}g"):
                if group.get("data-operation") == "cut":
                    for path in group.findall(f"{{{SVG}}}path"):
                        rings += _rings(path.get("d", ""))
                elif group.get("data-fold"):
                    for path in group.findall(f"{{{SVG}}}path"):
                        points = _points(path.get("d", ""))
                        if len(points) >= 2:
                            folds.append((LineString(points), group.get("data-fold"), path))
            if not rings:
                continue
            outline = Polygon(rings[0])
            if not outline.is_valid:
                outline = outline.buffer(0)
            shapes[g.get("id")] = PieceShape(g, outline, rings[1:], folds, origin)
        return shapes

    def _glues(self, pairs) -> list[Glue]:
        glues = []
        for pair in pairs.pairs:
            tabs = [e for e in pair.edges if e.side == "tab" and len(e.line) == 4]
            plain = [e for e in pair.edges if e.side == "edge" and len(e.line) == 4]
            if len(pair.edges) == 2 and len(tabs) == 1 and len(plain) == 1 and tabs[0].piece != plain[0].piece:
                glues.append(Glue(pair.number, tabs[0], plain[0]))
        return glues

    # ---- faces -----------------------------------------------------------------------------
    def _split(self, shape: PieceShape, tab_lines: list[LineString]) -> None:
        """Faces of the piece, and its body: the faces that are not a traced tab being replaced.
        A tab face lies against one of `tab_lines` and touches no other fold."""
        lines = [LineString(shape.outline.exterior.coords)] + [f for f, _, _ in shape.folds]
        faces = [f for f in polygonize(unary_union(lines)) if f.area > 0.01
                 and shape.outline.buffer(0.05).contains(f.representative_point())]
        shape.faces = faces
        tabs = []
        for line in tab_lines:
            near = [f for f in faces if f.boundary.buffer(0.3).intersection(line).length > 0.8 * line.length]
            rim = shape.outline.exterior.buffer(0.1)
            candidates = [f for f in near if f.boundary.difference(line.buffer(0.3)).difference(rim).length < 0.5]
            if len(near) >= 2 and candidates:
                tabs.append(min(candidates, key=lambda f: f.area))
        shape.old_tabs = tabs
        body = unary_union([f for f in faces if all(f is not t for t in tabs)])
        shape.body = _largest(body)

    # ---- sides -----------------------------------------------------------------------------
    def _choose_sides(self, glues: list[Glue], shapes: dict[str, PieceShape]) -> None:
        by_number = {(g.tab.piece, g.number): g for g in glues} | {(g.edge.piece, g.number): g for g in glues}
        for g in glues:
            twin_key = self.mirror_numbers.get((g.tab.piece, g.number))
            g.mirror = by_number.get(twin_key) if twin_key else None
            g.side = g.tab.piece
        # Centre seams (a piece glued to its own twin): alternate along the seam.
        seams = defaultdict(list)
        for g in glues:
            a, b = g.pieces
            if self.twin.get(a) == b:
                seams[tuple(sorted((a, b)))].append(g)
        for (a, b), seam in seams.items():
            ring = shapes[a].body.exterior
            seam.sort(key=lambda g: ring.project(shapes[a].line(g.tab if g.tab.piece == a else g.edge).centroid))
            first = sum(g.side == (a if i % 2 == 0 else b) for i, g in enumerate(seam)) >= len(seam) / 2
            for i, g in enumerate(seam):
                side = a if (i % 2 == 0) == first else b
                self.stats["centre seam flipped"] += side != g.side
                g.side = side
                g.mirror = None
        # Everywhere else, mirror: the first of two twin glues decides for both.
        for g in sorted(glues, key=lambda g: (min(g.pieces), g.number)):
            m = g.mirror
            if m is None or m is g or self.twin.get(g.side) is None:
                continue
            want = self.twin[g.side]
            if want in m.pieces and m.side != want:
                m.side = want
                self.stats["mirrored to twin"] += 1
            m.mirror = None   # decided

    # ---- geometry --------------------------------------------------------------------------
    def _tab(self, g: Glue, shapes: dict[str, PieceShape], placed: dict[str, list[Polygon]],
             depth: float) -> Polygon | None:
        on = g.side
        partner = g.edge.piece if on == g.tab.piece else g.tab.piece
        mine, theirs = shapes[on], shapes[partner]
        seg = _snap(mine.line(g.tab if on == g.tab.piece else g.edge), mine.body)
        other = _snap(theirs.line(g.edge if on == g.tab.piece else g.tab), theirs.body)
        face = _face(theirs, other)
        if face is None or seg.length < 1.0:
            return None
        (t0, t1), (s0, s1) = list(seg.coords), list(other.coords)
        inside = mine.body.buffer(-0.05)
        best = None
        for a, b in ((s0, s1), (s1, s0)):   # lay the partner face on the outside of the fold
            laid = _move(face, a, b, t0, t1)
            if not laid.representative_point().within(inside):
                overlap = laid.intersection(mine.body).area
                if best is None or overlap < best[0]:
                    best = (overlap, laid, _move(other, a, b, t0, t1))
        if best is None:
            return None
        _, laid, laid_seg = best
        out = _outward(seg, mine.body)
        cut = self.style.depth_mm / math.tan(math.radians(self.style.angle_deg))
        cut = min(cut, 0.45 * seg.length)
        p0, p1 = np.array(t0), np.array(t1)
        u = (p1 - p0) / seg.length
        d = self.style.depth_mm
        shape = Polygon([tuple(p0), tuple(p1), tuple(p1 - u * cut + out * d), tuple(p0 + u * cut + out * d)])
        if depth > self.style.depth_mm:   # a flap: the tab plus a paper doll's tongue, centred on the edge
            side, reach = min(self.style.flap_mm, seg.length), self.style.flap_length_mm
            a, b = p0 + u * (seg.length - side) / 2, p1 - u * (seg.length - side) / 2
            shape = shape.union(Polygon([tuple(a), tuple(b), tuple(b + out * reach), tuple(a + out * reach)]))
        # Clear of the partner face's other edges: its folds (the tab would have to bend)
        # and its cuts (the tab would show).
        edges = [LineString([c, n]) for c, n in zip(laid.exterior.coords, laid.exterior.coords[1:])]
        keep_off = unary_union([e for e in edges if e.distance(laid_seg.centroid) > 0.05
                                or abs(e.length - laid_seg.length) > 0.5]).buffer(self.style.margin_mm)
        shape = shape.intersection(laid).difference(keep_off)
        shape = shape.difference(mine.body).difference(
            unary_union(placed[on]).buffer(self.style.gap_mm) if placed[on] else Polygon())
        shape = _touching(shape, seg)
        if shape is None or shape.area < self.style.min_mm2:
            return None
        return _simplify(shape)

    def _flaps(self, glues, seams, length, shapes) -> set[int]:
        """Square flaps where they hold the most: on the longest glue of a seam between two big
        pieces, of a seam among the most loaded in the 3D model (stress.py), and at the stress
        points listed under `tabs: flaps:` (a piece id flaps the longest glue of each of its
        seams, a PDF number that glue). A twin follows its twin."""
        listed = set(self.style.flaps)
        listed |= {self.twin[p] for p in listed if p in self.twin}
        numbers = {n for n in listed if not n.startswith("piece-")}
        numbers |= {self.mirror_numbers[(p, n)][1] for g in glues for p in g.pieces
                    for n in [g.number] if n in numbers and (p, n) in self.mirror_numbers}
        stressed = self.stressed | {frozenset(self.twin.get(p) for p in key) for key in self.stressed
                                    if all(p in self.twin for p in key)}
        chosen = set()
        for pieces, seam in seams.items():
            longest = max(seam, key=lambda g: length[id(g)])
            big = all(shapes[p].body.area >= self.style.flap_area_mm2 for p in pieces) \
                and length[id(longest)] >= self.style.flap_edge_mm
            if big or pieces & listed or pieces in stressed:
                self.stats["stress flaps"] += pieces in stressed
                chosen.add(id(longest))
        return chosen | {id(g) for g in glues if g.number in numbers}

    # ---- the stage -------------------------------------------------------------------------
    def apply(self, root: ET.Element, pairs) -> dict[str, int]:
        shapes = self._shapes(root)
        glues = [g for g in self._glues(pairs) if g.tab.piece in shapes and g.edge.piece in shapes]
        tab_lines = defaultdict(list)
        for g in glues:
            line = shapes[g.tab.piece].line(g.tab)
            fold = min(shapes[g.tab.piece].folds, key=lambda f: f[0].hausdorff_distance(line), default=None)
            if fold and fold[0].hausdorff_distance(line) <= ON_LINE_MM:
                g.kind = fold[1]
                tab_lines[g.tab.piece].append(fold[0])
        glues = [g for g in glues if g.kind]
        material = {pid: s.element.get("data-material") for pid, s in shapes.items()}
        glues = [g for g in glues if not {material[p] for p in g.pieces} & set(self.style.keep)]
        for pid, shape in shapes.items():
            self._split(shape, tab_lines.get(pid, []))
        extras = self._extras(glues, shapes) if self.style.extra_flaps else []
        if not self.style.generate:
            glues = []
        # Only replace tabs whose traced face was found; the rest stay as traced.
        found = {pid: s.old_tabs for pid, s in shapes.items()}
        glues = [g for g in glues if any(t.boundary.buffer(0.3).intersection(shapes[g.tab.piece].line(g.tab)).length
                                         > 0.5 * shapes[g.tab.piece].line(g.tab).length for t in found[g.tab.piece])]
        self._choose_sides(glues, shapes)
        length = {id(g): shapes[g.tab.piece].line(g.tab).length for g in glues}
        seams = defaultdict(list)
        for g in glues:
            seams[frozenset(g.pieces)].append(g)
        flaps = self._flaps(glues, seams, length, shapes) if self.style.flap_mm else set()
        placed = defaultdict(list)
        if not self.style.generate:   # traced tabs stay, so extra flaps keep clear of them
            for pid, shape in shapes.items():
                placed[pid] += shape.old_tabs
        # Flaps first, then long edges: they need the room most.
        for g in sorted(glues, key=lambda g: (id(g) not in flaps, -length[id(g)])):
            depth = self.style.flap_mm if id(g) in flaps else self.style.depth_mm
            g.shape = self._tab(g, shapes, placed, depth)
            if g.shape is None and g.side != g.tab.piece:   # no room on the new side: back to the PDF's
                g.side = g.tab.piece
                g.shape = self._tab(g, shapes, placed, depth)
            self.stats["flaps"] += g.shape is not None and id(g) in flaps
            if g.shape is not None:
                placed[g.side].append(g.shape)
        added = []
        for g in sorted(extras, key=lambda g: -shapes[g.edge.piece].line(g.edge).length):
            g.shape = self._tab(g, shapes, placed, self.style.flap_mm)
            if g.shape is not None:
                placed[g.side].append(g.shape)
                added.append(g)
        self.stats["extra flaps"] = len(added)
        self._rewrite(shapes, glues, tab_lines, added)
        self.stats["tabs generated"] = sum(g.shape is not None for g in glues)
        # Tabs the partner face cut shallower than a tape strip (0.5 mm slack): worth a look.
        self.stats["tabs shallower than the tape"] = sum(
            g.shape is not None and _reach(g.shape, shapes[g.side].line(g.tab if g.side == g.tab.piece else g.edge))
            < self.style.depth_mm - 0.5 for g in glues)
        self.stats["tabs kept as traced"] = sum(g.shape is None for g in glues)
        self.stats["tabs moved to the other piece"] = sum(g.shape is not None and g.side != g.tab.piece for g in glues)
        return dict(self.stats)

    def _extras(self, glues: list[Glue], shapes: dict[str, PieceShape]) -> list[Glue]:
        """A flap on every plain glue edge of the `extra_flaps` pieces (and their twins), folded from
        that piece onto its partner's face, while the partner keeps its tab: both pieces then hold
        the seam, a paper doll's flap on one side and the tab on the other."""
        patterns = [p if p.startswith("piece-") else f"piece-{p}" for p in self.style.extra_flaps]
        listed = {pid for pid in shapes if any(fnmatch(pid, pattern) for pattern in patterns)}
        listed |= {self.twin[p] for p in listed if p in self.twin}
        return [Glue(g.number, g.tab, g.edge, g.kind, side=g.edge.piece) for g in glues if g.edge.piece in listed]

    def _rewrite(self, shapes: dict[str, PieceShape], glues: list[Glue], tab_lines, extras=()) -> None:
        gone = defaultdict(list)    # old tab faces removed, per piece
        new = defaultdict(list)     # (tab polygon, fold line, kind)
        for g in extras:            # the partner keeps its tab, the edge rows keep their sides
            seg = _snap(shapes[g.side].line(g.edge), shapes[g.side].body)
            base = g.shape.boundary.intersection(seg.buffer(0.01))
            base = LineString(base.coords) if isinstance(base, LineString) and base.length > 0.5 else seg
            new[g.side].append((g.shape, base, g.kind))
        for g in glues:
            if g.shape is None:
                continue
            tab_shape = shapes[g.tab.piece]
            line = tab_shape.line(g.tab)
            old = [t for t in tab_shape.old_tabs if t.boundary.buffer(0.3).intersection(line).length > 0.5 * line.length]
            gone[g.tab.piece] += old
            on = shapes[g.side]
            seg = _snap(on.line(g.tab if g.side == g.tab.piece else g.edge), on.body)
            base = g.shape.boundary.intersection(seg.buffer(0.01))   # a flap folds where it meets the edge
            base = LineString(base.coords) if isinstance(base, LineString) and base.length > 0.5 else seg
            new[g.side].append((g.shape, base, g.kind))
            if g.side != g.tab.piece:   # the edge rows trade sides; labels follow
                g.tab.side, g.edge.side = "edge", "tab"
        for pid, shape in shapes.items():
            if not gone[pid] and not new[pid]:
                continue
            keep = [f for f in shape.faces if all(f is not t for t in gone[pid])]
            outline = _largest(unary_union(keep + [t for t, _, _ in new[pid]]).buffer(0.001).buffer(-0.001))
            outline = _simplify(outline)
            ring = [(x, y) for x, y in outline.exterior.coords[:-1]]
            cut = next(g for g in shape.element.iter(f"{{{SVG}}}g") if g.get("data-operation") == "cut")
            paths = cut.findall(f"{{{SVG}}}path")
            paths[0].set("d", _d(ring) + "".join(" " + _d(h) for h in shape.holes))
            for extra in paths[1:]:
                cut.remove(extra)
            # Old tab folds whose tab is gone become cut edges (no fold).
            for fold, _, element in shape.folds:
                if any(fold.hausdorff_distance(t.boundary.intersection(fold.buffer(0.3))) < ON_LINE_MM
                       if not t.boundary.intersection(fold.buffer(0.3)).is_empty else False for t in gone[pid]) \
                        and fold.distance(outline.exterior) < 0.05 and fold.within(outline.exterior.buffer(0.05)):
                    for parent in shape.element.iter(f"{{{SVG}}}g"):
                        if element in list(parent):
                            parent.remove(element)
            for _, seg, kind in new[pid]:
                if any(f.hausdorff_distance(seg) < ON_LINE_MM for f, _, _ in shape.folds
                       if f.within(outline.buffer(0.05))):
                    continue   # the traced tab fold stays as it is
                group = next((g for g in shape.element.iter(f"{{{SVG}}}g") if g.get("data-fold") == kind), None)
                if group is None:
                    score = next(g for g in shape.element.iter(f"{{{SVG}}}g") if g.get("data-operation") == "score")
                    group = ET.SubElement(score, f"{{{SVG}}}g", {"data-fold": kind, "id": f"{pid}-{kind}"})
                attrs = {"d": _d(list(seg.coords), closed=False), "data-tab": "generated"}
                if paths[0].get("data-operation"):   # an export SVG marks every path
                    attrs.update({"data-operation": "score", "data-fold": kind})
                ET.SubElement(group, f"{{{SVG}}}path", attrs)


# ---- helpers -----------------------------------------------------------------------------------
def _points(d: str) -> list[tuple[float, float]]:
    v = [float(n) for n in NUMBER.findall(d)]
    return list(zip(v[::2], v[1::2]))


def _rings(d: str) -> list[list[tuple[float, float]]]:
    return [p for p in (_points(part) for part in re.split(r"[Mm]", d) if part.strip()) if len(p) >= 3]


def _d(points, closed: bool = True) -> str:
    body = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return f"M {body}" + (" Z" if closed else "")


def _largest(geometry) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    polygons = [g for g in getattr(geometry, "geoms", []) if isinstance(g, Polygon)]
    return max(polygons, key=lambda p: p.area) if polygons else Polygon()


def _touching(geometry, seg: LineString) -> Polygon | None:
    polygons = [geometry] if isinstance(geometry, Polygon) else \
        [g for g in getattr(geometry, "geoms", []) if isinstance(g, Polygon)]
    polygons = [p for p in polygons if not p.is_empty and p.distance(seg) < 0.01]
    return max(polygons, key=lambda p: p.area) if polygons else None


def _simplify(polygon: Polygon) -> Polygon:
    simple = polygon.simplify(0.02, preserve_topology=True)
    return simple if isinstance(simple, Polygon) and not simple.is_empty else polygon


def _snap(line: LineString, body: Polygon) -> LineString:
    """Ends of an edge-table line moved onto the nearest body corner within ON_LINE_MM."""
    corners = list(body.exterior.coords[:-1])
    ends = []
    for p in line.coords:
        best = min(corners, key=lambda c: math.dist(c, p), default=p)
        ends.append(best if math.dist(best, p) <= ON_LINE_MM else p)
    return LineString(ends)


def _face(shape: PieceShape, seg: LineString) -> Polygon | None:
    """The body face that has `seg` on its boundary."""
    near = [f for f in shape.faces if f.within(shape.body.buffer(0.05))
            and f.boundary.buffer(0.3).intersection(seg).length > 0.8 * seg.length]
    return max(near, key=lambda f: f.area) if near else None


def _move(geometry, a, b, c, d):
    """Rigid motion (no mirror) taking segment a-b onto c-d, matched at their midpoints."""
    angle = math.atan2(d[1] - c[1], d[0] - c[0]) - math.atan2(b[1] - a[1], b[0] - a[0])
    mid_ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    mid_cd = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)
    turned = affinity.rotate(geometry, angle, origin=Point(mid_ab), use_radians=True)
    return affinity.translate(turned, mid_cd[0] - mid_ab[0], mid_cd[1] - mid_ab[1])


def _reach(tab: Polygon, seg: LineString) -> float:
    """How far a tab reaches from its fold line."""
    return max(LineString(seg.coords).distance(Point(c)) for c in tab.exterior.coords)


def _outward(seg: LineString, body: Polygon) -> np.ndarray:
    (ax, ay), (bx, by) = seg.coords
    n = np.array([by - ay, ax - bx]) / seg.length
    mid = np.array([(ax + bx) / 2, (ay + by) / 2])
    return n if not Point(*(mid + 0.2 * n)).within(body) else -n


def main() -> int:
    """Add the YAML's `tabs: extra_flaps:` to an export that is already cut, in place: labels and
    every other piece stay as they are, so only the flapped pieces need a recut."""
    import argparse

    import yaml

    from .export import ExportContract, GluePairs
    parser = argparse.ArgumentParser(description="Add extra flaps to an existing export SVG.")
    parser.add_argument("--config", type=Path, required=True, help="project YAML with tabs: extra_flaps:")
    parser.add_argument("--svg", type=Path, help="export SVG (default: <folder>/<project>-export.svg)")
    args = parser.parse_args()
    folder, project = args.config.parent, args.config.stem
    raw = (yaml.safe_load(args.config.read_text(encoding="utf-8")).get("tabs") or {})
    style = TabStyle.from_config({**raw, "generate": False} if isinstance(raw, dict) else {})
    if style is None or not style.extra_flaps:
        parser.error("no `tabs: extra_flaps:` in the YAML")
    target = args.svg or folder / f"{project}-export.svg"
    tree = ET.parse(target)
    stats = Retabber(style, folder / "mirror-mismatches.csv").apply(tree.getroot(), GluePairs.read(folder / "edge-numbers.csv"))
    ET.indent(tree, space="")
    tree.write(target, encoding="UTF-8", xml_declaration=True)
    problems = ExportContract(target).errors()
    print(f"{target}: {stats.get('extra flaps', 0)} extra flaps")
    for problem in problems:
        print(f"contract: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
