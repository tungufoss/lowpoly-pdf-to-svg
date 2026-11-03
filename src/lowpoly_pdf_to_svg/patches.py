"""Backing patches: a second layer glued inside across a weak seam.

    python -m src.lowpoly_pdf_to_svg.patches --config data/bowser/bowser.yaml \
        --pieces piece-11-04 piece-11-05 --short-mm 15

(or leave out --pieces and list groups under `patches:` in the YAML, each with `pieces:`
(ids or patterns such as piece-24-*), `short_mm:` and `joins: true` to patch only the seams
joining the group to the rest, e.g. teeth to the jaw).

For every glue pair on the listed pieces (and their twins) whose edge, or its mirror edge on
the twins, is shorter than `--short-mm`, the two
faces that meet at the seam are laid flat side by side and cut as one patch, reaching at
most `--reach-mm` (one tape width) either side of the seam and kept `--margin-mm` inside
the faces' other edges, so the patch never has to bend anywhere but at the seam. It sits
inside the model, under the pieces, so its score line is the seam's fold inverted (Helga:
a second piece can overlap if the scoring is inverted). Each patch carries the seam's
export label. Patches are packed in rows onto Letter sheets, one cardstock per sheet, in one SVG,
<project>-patches.svg, with Cut, Score and Draw layers like the export.
"""

from __future__ import annotations

import argparse
import csv
import math
from fnmatch import fnmatch
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .export import DEFAULT_COLORS, FALLBACK_COLOR, GluePairs, SVG
from .retab import Retabber, TabStyle, _face, _largest, _move, _snap

INVERT = {"mountain": "valley", "valley": "mountain"}
SHEET = (215.9, 279.4)   # Letter, mm
GAP = 3.0


class PatchMaker:
    def __init__(self, svg: Path, pairs: GluePairs, labels: dict[tuple[str, frozenset], str],
                 reach_mm: float = 15.0, margin_mm: float = 0.8):
        self.root = ET.parse(svg).getroot()
        self.pairs, self.labels = pairs, labels
        self.reach, self.margin = reach_mm, margin_mm
        self.retab = Retabber(TabStyle(), svg.with_name("mirror-mismatches.csv"))
        self.shapes = self.retab._shapes(self.root)
        self.glues = [g for g in self.retab._glues(pairs) if g.tab.piece in self.shapes and g.edge.piece in self.shapes]
        tab_lines = {}
        for g in self.glues:
            shape = self.shapes[g.tab.piece]
            line = shape.line(g.tab)
            fold = min(shape.folds, key=lambda f: f[0].hausdorff_distance(line), default=None)
            if fold and fold[0].hausdorff_distance(line) <= 1.0:
                g.kind = fold[1]
                tab_lines.setdefault(g.tab.piece, []).append(fold[0])
        for pid, shape in self.shapes.items():
            self.retab._split(shape, tab_lines.get(pid, []))

    def patches(self, pieces: set[str], short_mm: float, joins: bool = False) -> list[dict]:
        """Patches for the glue edges on `pieces` shorter than `short_mm`; with `joins`, only
        the seams that join the group to the rest of the model (teeth to the jaw)."""
        # A seam is patched when it or its mirror seam on the twins is short, so twins match.
        pieces = pieces | {self.retab.twin[p] for p in pieces if p in self.retab.twin}
        by_number = {(p, g.number): g for g in self.glues for p in g.pieces}
        wanted = set()
        for g in self.glues:
            inside = set(g.pieces) & pieces
            if joins and len(inside) == 2:
                continue
            if g.kind and inside and self.shapes[g.edge.piece].line(g.edge).length < short_mm:
                wanted.add(id(g))
                for p in g.pieces:
                    twin = by_number.get(self.retab.mirror_numbers.get((p, g.number)))
                    if twin is not None and twin.kind:
                        wanted.add(id(twin))
        out = []
        for g in self.glues:
            if id(g) not in wanted:
                continue
            plain, tab = self.shapes[g.edge.piece], self.shapes[g.tab.piece]
            seg, other = _snap(plain.line(g.edge), plain.body), _snap(tab.line(g.tab), tab.body)
            here, there = _face(plain, seg), _face(tab, other)
            if here is None or there is None:
                continue
            (a, b), (c, d) = list(seg.coords), list(other.coords)
            laid = min((_move(there, p, q, a, b) for p, q in ((c, d), (d, c))),
                       key=lambda f: f.intersection(plain.body).area)
            shape = self._clip(here, seg).union(self._clip(laid, seg)).buffer(0.001).buffer(-0.001)
            shape = _largest(shape)
            if shape.is_empty or shape.area < 20:
                continue
            fold = shape.intersection(seg.buffer(0.01))
            fold = LineString(fold.coords) if isinstance(fold, LineString) and fold.length > 0.5 else seg
            out.append({"shape": shape, "fold": fold, "kind": INVERT.get(g.kind, g.kind),
                        "label": self.labels.get((g.number, frozenset(g.pieces)), g.number),
                        "material": (plain if g.edge.piece in pieces else tab).element.get("data-material"), "seam": f"{g.edge.piece} / {g.tab.piece}",
                        "length": seg.length})
        return out

    def _clip(self, face: Polygon, seg: LineString) -> Polygon:
        """The face within `reach` of the seam, `margin` clear of its other edges."""
        ring = list(face.exterior.coords)
        edges = [LineString([p, q]) for p, q in zip(ring, ring[1:])]
        others = unary_union([e for e in edges if e.hausdorff_distance(seg) > 0.3])
        return face.intersection(seg.buffer(self.reach, cap_style="flat")).difference(others.buffer(self.margin))


def pack(patches: list[dict]) -> list[list[dict]]:
    """Rows on Letter sheets, each patch turned to its smallest upright box."""
    sheets, x, y, row = [[]], GAP, GAP, 0.0
    for p in sorted(patches, key=lambda p: -p["shape"].area):
        best = None
        for k in range(36):
            turned = affinity.rotate(p["shape"], k * 5, origin=p["shape"].centroid)
            x0, y0, x1, y1 = turned.bounds
            if best is None or (y1 - y0, x1 - x0) < best[1:]:
                best = (k * 5, y1 - y0, x1 - x0)
        angle, h, w = best
        if x + w > SHEET[0] - GAP:
            x, y, row = GAP, y + row + GAP, 0.0
        if y + h > SHEET[1] - GAP:
            sheets.append([])
            x, y, row = GAP, GAP, 0.0
        turned = affinity.rotate(p["shape"], angle, origin=p["shape"].centroid)
        x0, y0, _, _ = turned.bounds
        p["placed"] = affinity.translate(turned, x - x0, y - y0)
        p["placed_fold"] = affinity.translate(affinity.rotate(p["fold"], angle, origin=p["shape"].centroid), x - x0, y - y0)
        sheets[-1].append(p)
        x, row = x + w + GAP, max(row, h)
    return sheets


def write(sheets: list[list[dict]], colors: dict[str, str], target: Path) -> None:
    width, height = SHEET[0], SHEET[1] * len(sheets) + 10 * (len(sheets) - 1)
    root = ET.Element(f"{{{SVG}}}svg", {"width": f"{width}mm", "height": f"{height}mm",
                                       "viewBox": f"0 0 {width:.1f} {height:.1f}"})
    ET.SubElement(root, f"{{{SVG}}}title").text = "Backing patches"
    for n, sheet in enumerate(sheets):
        oy = n * (SHEET[1] + 10)
        page = ET.SubElement(root, f"{{{SVG}}}g", {"id": f"sheet-{n + 1}", "transform": f"translate(0 {oy:.1f})"})
        ET.SubElement(page, f"{{{SVG}}}rect", {"width": f"{SHEET[0]}", "height": f"{SHEET[1]}",
                                               "style": "fill:none;stroke:#bbbbbb;stroke-width:0.2"})
        for i, p in enumerate(sheet):
            color = colors.get(p["material"], FALLBACK_COLOR)
            g = ET.SubElement(page, f"{{{SVG}}}g", {"id": f"patch-{n + 1}-{i + 1}", "data-label": p["label"],
                                                    "data-seam": p["seam"], "data-material": p["material"] or ""})
            ring = " ".join(f"{x:.3f},{y:.3f}" for x, y in list(p["placed"].exterior.coords)[:-1])
            ET.SubElement(g, f"{{{SVG}}}path", {"d": f"M {ring} Z", "data-operation": "cut",
                                                "style": f"fill:{color};stroke:#000000;stroke-width:0.25"})
            (ax, ay), (bx, by) = list(p["placed_fold"].coords)[0], list(p["placed_fold"].coords)[-1]
            dash = "2.5,0.8,0.4,0.8" if p["kind"] == "mountain" else "1.5,1"
            ET.SubElement(g, f"{{{SVG}}}path", {"d": f"M {ax:.3f},{ay:.3f} {bx:.3f},{by:.3f}", "data-operation": "score",
                                                "data-fold": p["kind"],
                                                "style": f"fill:none;stroke:#1f5fbf;stroke-width:0.25;stroke-dasharray:{dash}"})
            c = p["placed"].representative_point()
            text = ET.SubElement(g, f"{{{SVG}}}text", {"x": f"{c.x:.2f}", "y": f"{c.y:.2f}", "font-size": "3",
                                                       "text-anchor": "middle", "dominant-baseline": "central",
                                                       "data-operation": "draw",
                                                       "style": "font-family:Arial,sans-serif;font-weight:bold;fill:#8e24aa"})
            text.text = p["label"]
    ET.indent(root, space="")
    ET.ElementTree(root).write(target, encoding="UTF-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backing patches for short, weak seams.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pieces", nargs="+", help="piece ids whose short seams get patches "
                                                     "(default: `patches: pieces:` in the YAML)")
    parser.add_argument("--short-mm", type=float, help="patch glue edges shorter than this (default 15, or `patches: short_mm:`)")
    parser.add_argument("--reach-mm", type=float, default=15.0, help="how far a patch reaches either side of the seam")
    parser.add_argument("--margin-mm", type=float, default=0.8, help="clearance from the faces' other edges")
    args = parser.parse_args()
    folder, project = args.config.parent, args.config.stem
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    colors = {str(m["id"]): str(m.get("color") or DEFAULT_COLORS.get(str(m["id"]).lower(), FALLBACK_COLOR))
              for m in raw["materials"]}
    labels = {}
    if (folder / f"{project}-labels.csv").is_file():
        with (folder / f"{project}-labels.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                labels[(row["pdf_number"], frozenset(row["pieces"].split()))] = row["label"]
    groups = raw.get("patches") or []
    groups = [groups] if isinstance(groups, dict) else groups
    if args.pieces:
        groups = [{"pieces": args.pieces, "short_mm": args.short_mm or 15.0}]
    if not groups:
        parser.error("no pieces: pass --pieces or list them under `patches:` in the YAML")
    maker = PatchMaker(folder / f"{project}.svg", GluePairs.read(folder / "edge-numbers.csv"), labels,
                       args.reach_mm, args.margin_mm)
    found, seen = [], set()
    for group in groups:
        patterns = [str(p) if str(p).startswith("piece-") else f"piece-{p}" for p in group.get("pieces", [])]
        pieces = {pid for pid in maker.shapes if any(fnmatch(pid, pattern) for pattern in patterns)}
        for patch in maker.patches(pieces, args.short_mm or float(group.get("short_mm", 15.0)),
                                   bool(group.get("joins", False))):
            if patch["seam"] + patch["label"] not in seen:
                seen.add(patch["seam"] + patch["label"])
                found.append(patch)
    sheets = [sheet for material in sorted({p["material"] or "" for p in found})   # one cardstock per sheet
              for sheet in pack([p for p in found if (p["material"] or "") == material])]
    target = folder / f"{project}-patches.svg"
    write(sheets, colors, target)
    print(f"{target}: {len(found)} patches on {len(sheets)} sheet(s)")
    for p in sorted(found, key=lambda p: p["label"]):
        print(f"  {p['label']:>4}  {p['seam']}  seam {p['length']:.1f} mm, fold {p['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
