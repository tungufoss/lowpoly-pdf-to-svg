"""Stage 2b': edge numbers fixed by hand.

data/<project>/number-corrections.csv lists, per number, the piece, where the
number sits on its page (mm) and what it should say. The fix page (`make fix`)
writes this file; every build applies it after OCR, so a number corrected once
stays corrected. A row is matched to the label of that piece nearest its position.
"""

from __future__ import annotations

import collections
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .model import Fold, FoldKind, Label, PageLayout

MATCH_MM = 2.0
NOT_A_NUMBER = "x"     # the number check page's answer for marks that are no number (a fan of fold ends)
COLUMNS = ["piece", "x_mm", "y_mm", "read", "number"]


@dataclass
class Correction:
    piece: str
    x_mm: float
    y_mm: float
    read: str
    number: str


class Corrections:
    def __init__(self, rows: list[Correction], by_hand: bool = True):
        self.rows, self.by_hand = rows, by_hand

    @classmethod
    def load(cls, path: Path) -> "Corrections":
        if not path.is_file():
            return cls([])
        with path.open(encoding="utf-8", newline="") as handle:
            return cls([Correction(r["piece"], float(r["x_mm"]), float(r["y_mm"]), r.get("read", ""), r["number"].strip())
                        for r in csv.DictReader(handle) if r.get("number", "").strip()])

    def match(self, layouts: list[tuple[PageLayout, float]]) -> dict[int, str]:
        """id(label) -> corrected text, for the labels these corrections point at."""
        found = {}
        pieces = {piece.id: (piece, mm) for layout, mm in layouts for piece in layout.pieces}
        for row in self.rows:
            label = self._nearest(pieces, row)
            if label is not None and row.number != NOT_A_NUMBER:
                found[id(label)] = row.number
        return found

    @staticmethod
    def _nearest(pieces, row: Correction):
        piece, mm = pieces.get(row.piece, (None, 0.0))
        if piece is None or not piece.labels:
            return None
        distance = lambda l: (l.center[0] * mm - row.x_mm) ** 2 + (l.center[1] * mm - row.y_mm) ** 2
        label = min(piece.labels, key=distance)
        return label if distance(label) <= MATCH_MM ** 2 else None

    def apply(self, layouts: list[tuple[PageLayout, float]]) -> tuple[int, list[Correction]]:
        """Set each corrected label's text; returns (applied, rows that matched no label)."""
        pieces = {piece.id: (piece, mm) for layout, mm in layouts for piece in layout.pieces}
        applied, missed = 0, []
        for row in self.rows:
            label = self._nearest(pieces, row)
            if label is None:
                missed.append(row)
                continue
            if row.number == NOT_A_NUMBER:
                pieces[row.piece][0].labels.remove(label)
                applied += 1
                continue
            label.text, label.confident, label.conf = row.number, True, 100.0
            label.corrected = label.corrected or self.by_hand
            applied += 1
        if applied:
            counts = collections.Counter(l.text for layout, _ in layouts for p in layout.pieces for l in p.labels)
            for layout, _ in layouts:
                for p in layout.pieces:
                    for label in p.labels:
                        label.paired = counts[label.text] == 2
        return applied, missed


LINE_COLUMNS = ["piece", "x_mm", "y_mm", "number", "kind", "ax_mm", "ay_mm", "bx_mm", "by_mm"]


@dataclass
class LineFix:
    """The line a number marks, drawn by hand on the number check page: two points on the page (mm),
    and whether it is a cut edge or a fold (a fold means this side carries the glue tab)."""
    piece: str
    x_mm: float
    y_mm: float
    kind: str
    a: tuple[float, float]
    b: tuple[float, float]


class LineFixes:
    """data/<project>/number-lines.csv: hand-drawn lines that replace what tracing found for a number."""

    def __init__(self, rows: list[LineFix]):
        self.rows = rows

    @classmethod
    def load(cls, path: Path) -> "LineFixes":
        if not path.is_file():
            return cls([])
        with path.open(encoding="utf-8", newline="") as handle:
            return cls([LineFix(r["piece"], float(r["x_mm"]), float(r["y_mm"]), r["kind"],
                                (float(r["ax_mm"]), float(r["ay_mm"])), (float(r["bx_mm"]), float(r["by_mm"])))
                        for r in csv.DictReader(handle) if r.get("kind") in ("cut", "fold")])

    def apply(self, layouts: list[tuple[PageLayout, float]]) -> tuple[int, list[LineFix]]:
        """Set each matched label's line; returns (applied, rows that matched no label)."""
        pieces = {piece.id: (piece, mm) for layout, mm in layouts for piece in layout.pieces}
        applied, missed = 0, []
        for row in self.rows:
            label = Corrections._nearest(pieces, row)
            if label is None:
                missed.append(row)
                continue
            mm = pieces[row.piece][1]
            a, b = (row.a[0] / mm, row.a[1] / mm), (row.b[0] / mm, row.b[1] / mm)
            label.edge = (a, b)
            # Where the line lies decides cut or fold, not the label it was drawn with: on a fold it
            # is that fold (the number sits on the tab side), along the outline it is a cut edge.
            piece = pieces[row.piece][0]
            label.tab = row.kind == "fold"
            if _on_fold(piece, a, b, FOLD_MM / mm):
                label.tab = True
            elif _on_outline(piece, a, b, FOLD_MM / mm):
                label.tab = False
            applied += 1
        return applied, missed


PIECE_COLUMNS = ["piece", "action", "kind", "ax_mm", "ay_mm", "bx_mm", "by_mm", "note"]
MERGE_MM = 0.8         # corners and fold ends closer than this are one point
ON_LINE_MM = 0.5       # a point this close to a fold, away from its ends, splits it there
FOLD_MM = 1.5          # a dropped fold is the traced fold whose two ends lie this close to the row's


@dataclass
class PieceFix:
    """One answer from the piece review: `ok` (the piece was looked at), `add` (a fold the
    trace missed, with its kind), `drop` (a traced fold that is no fold) or `move` (an outline
    corner dragged from a to b, taking the folds that end there along), `drop-point` (remove an
    outline vertex and project folds ending there onto the joined edge), points in page mm.
    A `number` row adds a number the reader missed at a (its text in `kind`).
    A `note` row carries free text for edits the page cannot make; the build leaves it alone."""
    piece: str
    action: str
    kind: str = ""
    a: tuple[float, float] = (0.0, 0.0)
    b: tuple[float, float] = (0.0, 0.0)


class PieceFixes:
    """data/<project>/piece-edits.csv: folds added or dropped and corners moved by hand, piece by piece."""

    def __init__(self, rows: list[PieceFix]):
        self.rows = rows

    @classmethod
    def load(cls, path: Path) -> "PieceFixes":
        if not path.is_file():
            return cls([])
        number = lambda v: float(v) if v else 0.0
        with path.open(encoding="utf-8", newline="") as handle:
            return cls([PieceFix(r["piece"], r["action"], r.get("kind", ""),
                                 (number(r["ax_mm"]), number(r["ay_mm"])), (number(r["bx_mm"]), number(r["by_mm"])))
                        for r in csv.DictReader(handle) if r.get("action") in ("ok", "add", "drop", "move", "drop-point", "number")])

    def apply(self, layouts: list[tuple[PageLayout, float]], kind_of=None) -> tuple[set[str], list[PieceFix]]:
        """Move corners, add and drop folds; returns (ids of pieces changed, rows that matched nothing).
        kind_of(piece id, a, b) reads an added fold's style off the scan (None where no fold is
        printed); the style picked by hand counts only there."""
        pieces = {piece.id: (piece, mm) for layout, mm in layouts for piece in layout.pieces}
        changed, missed = set(), []
        for row in self.rows:
            if row.action == "ok":
                continue
            piece, mm = pieces.get(row.piece, (None, 0.0))
            if piece is None:
                missed.append(row)
                continue
            a, b = (row.a[0] / mm, row.a[1] / mm), (row.b[0] / mm, row.b[1] / mm)
            near = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= (FOLD_MM / mm) ** 2
            if row.action == "move":
                corners = [i for i, p in enumerate(piece.outline) if near(p, a)]
                if not corners:
                    missed.append(row)
                    continue
                moved = [tuple(piece.outline[corners[0]])]
                piece.outline[corners[0]] = b
                for fold in piece.folds:
                    moved += [tuple(e) for e in (fold.a, fold.b) if near(e, a)]
                    fold.a, fold.b = (b if near(fold.a, a) else fold.a), (b if near(fold.b, a) else fold.b)
                # a fold running through a point that moved bends with it, so it still splits there
                through = [f for f in piece.folds if not near(f.a, b) and not near(f.b, b)
                           and any(_through(f.a, f.b, e, ON_LINE_MM / mm) for e in moved)]
                for fold in through:
                    piece.folds.remove(fold)
                    piece.folds += [Fold(fold.a, b, fold.kind), Fold(b, fold.b, fold.kind)]
            elif row.action == "drop-point":
                corners = [i for i, p in enumerate(piece.outline) if near(p, a)]
                if not corners or len(piece.outline) <= 3:
                    missed.append(row)
                    continue
                index = corners[0]
                removed = tuple(piece.outline[index])
                previous = tuple(piece.outline[index - 1])
                following = tuple(piece.outline[(index + 1) % len(piece.outline)])

                def project(p):
                    dx, dy = following[0] - previous[0], following[1] - previous[1]
                    length2 = dx * dx + dy * dy
                    if not length2:
                        return previous
                    t = max(0.0, min(1.0, ((p[0] - previous[0]) * dx + (p[1] - previous[1]) * dy) / length2))
                    return (previous[0] + t * dx, previous[1] + t * dy)

                for fold in piece.folds:
                    if near(fold.a, removed):
                        fold.a = project(fold.a)
                    if near(fold.b, removed):
                        fold.b = project(fold.b)
                piece.outline.pop(index)
            elif row.action == "number":
                height = 2.0 / mm   # printed digits are about 2 mm tall
                box = (round(a[0] - height), round(a[1] - height / 2), round(2 * height), round(height))
                piece.labels.append(Label(box, a, 0.0, height, np.zeros((1, 1), np.uint8), text=row.kind,
                                          confident=True, corrected=True, elongation=1.0))
            elif row.action == "add":
                if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= (FOLD_MM / mm) ** 2:
                    continue   # both ends on one point: a slip, not a fold
                printed = kind_of(piece.id, a, b) if kind_of else None
                piece.folds.append(Fold(a, b, printed or FoldKind(row.kind or "valley")))
            else:
                # only the closest match goes: a fold added by hand on top of a traced one survives
                gap = lambda f: min(math.dist(f.a, a) + math.dist(f.b, b), math.dist(f.a, b) + math.dist(f.b, a))
                hits = [f for f in piece.folds
                        if (near(f.a, a) and near(f.b, b)) or (near(f.a, b) and near(f.b, a))]
                if not hits:
                    missed.append(row)
                    continue
                piece.folds.remove(min(hits, key=gap))
            changed.add(piece.id)
        for piece_id, (piece, mm) in pieces.items():
            before = (len(piece.outline), len(piece.folds))
            self.tidy(piece, mm)
            if (len(piece.outline), len(piece.folds)) != before:
                changed.add(piece_id)
        return changed, missed

    @staticmethod
    def tidy(piece, mm: float) -> None:
        """Clean a piece after hand edits: corners and fold ends that (nearly) coincide become one
        point, a fold with another point on it is split there, and folds that then repeat each
        other or run along the cut are dropped, so no two lines overlap."""
        merge, on_line = MERGE_MM / mm, ON_LINE_MM / mm
        close = lambda p, q: math.dist(p, q) <= merge
        outline = []
        for p in piece.outline:
            if not outline or not close(p, outline[-1]):
                outline.append(tuple(p))
        while len(outline) > 3 and close(outline[0], outline[-1]):
            outline.pop()
        piece.outline = outline
        points = list(outline)
        snap = lambda p: next((q for q in points if close(p, q)), None) or (points.append(tuple(p)) or tuple(p))
        folds = [(snap(f.a), snap(f.b), f) for f in piece.folds]
        parts = []
        for a, b, fold in folds:
            if a == b:
                continue
            length = math.dist(a, b)
            u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            cuts = sorted({t for p in points
                           for t in [(p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]]
                           if merge < t < length - merge and abs((p[0] - a[0]) * u[1] - (p[1] - a[1]) * u[0]) <= on_line})
            stops = [a] + [next(p for p in points if abs((p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1] - t) < 1e-6
                                and abs((p[0] - a[0]) * u[1] - (p[1] - a[1]) * u[0]) <= on_line) for t in cuts] + [b]
            parts += [(p, q, fold.kind) for p, q in zip(stops, stops[1:])]
        kept, seen = [], set()
        n = len(outline)
        sides = {frozenset((outline[i], outline[(i + 1) % n])) for i in range(n)}
        for p, q, kind in parts:
            key = frozenset((p, q))
            if key in seen or key in sides:
                continue
            seen.add(key)
            kept.append(Fold(p, q, kind))
        # a fold that two others already cover by way of a point near it (a slight bend) goes too
        ends = {e for f in kept for e in (f.a, f.b)}
        covered = lambda f: any(frozenset((f.a, e)) in seen and frozenset((e, f.b)) in seen
                                and math.dist(f.a, e) + math.dist(e, f.b) - math.dist(f.a, f.b) <= on_line
                                for e in ends if e not in (f.a, f.b))
        piece.folds = [f for f in kept if not covered(f)]


def _through(a, b, p, reach: float) -> bool:
    """p lies on the segment a-b, away from its ends, within reach."""
    length = math.dist(a, b)
    if length == 0:
        return False
    t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / length
    side = abs((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) / length
    return reach < t < length - reach and side <= reach


def _on_fold(piece, a, b, reach: float) -> bool:
    """Does a fold of the piece run from a to b (either way round)?"""
    on = lambda p, q: math.dist(p, q) <= reach
    return any((on(f.a, a) and on(f.b, b)) or (on(f.a, b) and on(f.b, a)) for f in piece.folds)


def _on_outline(piece, a, b, reach: float) -> bool:
    """Do both ends and the middle of the line lie on the cut outline?"""
    outline = np.asarray(piece.outline, np.float32).reshape(-1, 1, 2)
    middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return all(abs(cv2.pointPolygonTest(outline, (float(p[0]), float(p[1])), True)) <= reach for p in (a, b, middle))
