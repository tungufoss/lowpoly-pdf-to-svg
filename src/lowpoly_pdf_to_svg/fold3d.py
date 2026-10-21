"""Stage 5 (prototype): fold the flat pattern back into the 3D low-poly mask.

Reads the traced review SVG (cut outlines, mountain/valley folds, edge numbers),
splits every piece into flat faces along its folds, glues each plain edge to the
tab fold carrying the same number, and solves for 3D vertex positions that keep
every face rigid, close every glued edge and bend each fold the printed way.

Where the model will not close, the glued edge stays open in the result: the
gap in mm is listed per number, which points at a misread number or a mis-traced
line. All lengths are millimetres.

    python -m src.lowpoly_pdf_to_svg.fold3d --config data/example/example.yaml
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SVG = "{http://www.w3.org/2000/svg}"
INKSCAPE = "{http://www.inkscape.org/namespaces/inkscape}"
LABEL_REACH_MM = 3.0     # as detect.py: a number further than this from every line marks none
LENGTH_TOLERANCE_MM = 1.0  # as edges.py: a clean pair's two lines agree this well
SNAP_MM = 0.4            # fold ends and outline corners closer than this are one vertex

Point = tuple[float, float]


# -- flat pattern -----------------------------------------------------------------

@dataclass
class Label:
    text: str
    centre: np.ndarray
    angle: float
    trusted: bool
    edge: tuple[int, int] | None = None   # vertex ids of the marked line, in face order
    tab: bool = False


@dataclass
class Face:
    id: int
    piece: str
    vertices: list[int]       # counter-clockwise as printed (y up), vertex ids within the piece
    tab: bool = False         # a glue tab: folded under, not part of the surface


@dataclass
class FlatPiece:
    id: str
    material: str
    outline: list[Point]
    folds: list[tuple[Point, Point, str]]       # a, b, mountain|valley
    labels: list[Label] = field(default_factory=list)
    points: list[np.ndarray] = field(default_factory=list)   # vertex id -> 2D position
    faces: list[Face] = field(default_factory=list)
    fold_kind: dict[frozenset, str] = field(default_factory=dict)   # vertex pair -> fold kind
    page_origin: tuple[float, float] = (0.0, 0.0)                   # its page's corner in document mm

    # vertices -------------------------------------------------------------------
    def vertex(self, p, snap: float = SNAP_MM) -> int:
        p = np.asarray(p, np.float64)
        for i, q in enumerate(self.points):
            if np.hypot(*(p - q)) < snap:
                return i
        self.points.append(p)
        return len(self.points) - 1

    def split_faces(self) -> None:
        """Cut the outline along every fold into flat faces. A fold may end on a corner,
        part-way along a cut edge, or on another fold."""
        from shapely.geometry import LineString, Polygon
        from shapely.ops import polygonize, unary_union

        outline = Polygon(self.outline).buffer(0)
        lines = [LineString(outline.exterior.coords)]
        for a, b, _ in self.folds:
            a, b = np.asarray(a), np.asarray(b)
            u = (b - a) / max(np.hypot(*(b - a)), 1e-9)
            lines.append(LineString([a - u * SNAP_MM, b + u * SNAP_MM]))  # reach the cut line
        for corner in outline.exterior.coords[:-1]:
            self.vertex(corner)
        for a, b, _ in self.folds:
            self.vertex(a), self.vertex(b)
        for poly in polygonize(unary_union(lines)):
            if poly.area < 0.5 or not outline.buffer(0.05).contains(poly.representative_point()):
                continue
            ring = [self.vertex(p) for p in poly.exterior.coords[:-1]]
            ring = [v for i, v in enumerate(ring) if v != ring[i - 1]]
            if len(ring) < 3:
                continue
            if _signed_area([self.points[v] for v in ring]) < 0:   # svg y runs down: make ccw as seen
                ring.reverse()
            self.faces.append(Face(-1, self.id, ring))
        for a, b, kind in self.folds:
            self.fold_kind[frozenset((self.vertex(a), self.vertex(b)))] = kind

    def face_edges(self, face: Face):
        n = len(face.vertices)
        return [(face.vertices[i], face.vertices[(i + 1) % n]) for i in range(n)]

    def mark_labels(self) -> None:
        """As detect.py: each number marks the nearest line it runs along, a cut edge
        (plain side) or a fold (tab side)."""
        edges: dict[frozenset, tuple[int, int]] = {}
        count = collections.Counter()
        for face in self.faces:
            for a, b in self.face_edges(face):
                key = frozenset((a, b))
                count[key] += 1
                edges.setdefault(key, (a, b))
        for label in self.labels:
            reading = np.array([math.cos(math.radians(label.angle)), math.sin(math.radians(label.angle))])
            best, best_score = None, LABEL_REACH_MM
            for key, (a, b) in edges.items():
                pa, pb = self.points[a], self.points[b]
                length = np.hypot(*(pb - pa))
                if length < 1e-6:
                    continue
                u = (pb - pa) / length
                along = float((label.centre - pa) @ u)
                if not 0 <= along <= length:
                    continue
                across = abs(float((label.centre - pa) @ np.array([-u[1], u[0]])))
                score = across + (0 if abs(float(reading @ u)) > math.cos(math.radians(25)) else 2)
                if score < best_score:
                    best, best_score = key, score
            if best is not None:
                label.edge, label.tab = edges[best], count[best] == 2

    def mark_tabs(self) -> None:
        """A tab is the face beyond a numbered tab fold whose other sides are all cut."""
        shared = collections.Counter(frozenset(e) for f in self.faces for e in self.face_edges(f))
        for label in self.labels:
            if not label.tab:
                continue
            key = frozenset(label.edge)
            sides = [f for f in self.faces if any(frozenset(e) == key for e in self.face_edges(f))]
            outer = [f for f in sides if sum(shared[frozenset(e)] == 2 for e in self.face_edges(f)) == 1]
            if len(outer) == 1 and len(sides) == 2:
                outer[0].tab = True
            elif len(outer) == 2:   # a lone two-face piece: the smaller one is the tab
                min(outer, key=lambda f: abs(_signed_area([self.points[v] for v in f.vertices]))).tab = True

    def length(self, a: int, b: int) -> float:
        return float(np.hypot(*(self.points[a] - self.points[b])))


def _signed_area(points) -> float:
    pts = np.asarray(points)
    x, y = pts[:, 0], -pts[:, 1]   # y up
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class PatternReader:
    """Pieces as the review SVG draws them, in document millimetres."""

    ROTATE = re.compile(r"rotate\(([-\d.]+)")

    def read(self, path: Path) -> list[FlatPiece]:
        root = ET.parse(path).getroot()
        origins = {page.get("id"): (float(page.get("x")), float(page.get("y")))
                   for page in root.iter(f"{INKSCAPE}page")}
        pieces = []
        for group in root.iter(f"{SVG}g"):
            pid = group.get("id", "")
            if not re.fullmatch(r"piece-\d\d-\d\d", pid):
                continue
            cut = group.find(f"{SVG}g[@id='{pid}-cut']")
            paths = [self._points(p.get("d")) for p in cut.iter(f"{SVG}path")]
            piece = FlatPiece(pid, group.get("data-material", ""), paths[0], [],
                              page_origin=origins.get(f"page-{pid[6:8]}", (0.0, 0.0)))
            for kind in ("mountain", "valley"):
                for p in group.iter(f"{SVG}g"):
                    if p.get("id") == f"{pid}-{kind}":
                        piece.folds += [(*self._points(q.get("d"))[:2], kind) for q in p.iter(f"{SVG}path")]
            draw = group.find(f"{SVG}g[@id='{pid}-draw']")
            for t in draw.iter(f"{SVG}text"):
                angle = float(self.ROTATE.search(t.get("transform", "rotate(0")).group(1))
                piece.labels.append(Label((t.text or "").strip(), np.array([float(t.get("x")), float(t.get("y"))]),
                                          angle, t.get("data-review") != "ocr"))
            pieces.append(piece)
        return pieces

    @staticmethod
    def _points(d: str) -> list[Point]:
        nums = [tuple(map(float, xy.split(","))) for xy in re.findall(r"[-\d.]+,[-\d.]+", d)]
        return nums


# -- glue -------------------------------------------------------------------------

@dataclass
class Glue:
    """Number `number` glues `piece` edge a->b (plain side) onto `other` fold c->d (tab side).
    Both are taken in their face's printed counter-clockwise order, so a meets d and b meets c."""
    number: str
    piece: str
    a: int
    b: int
    other: str
    c: int
    d: int
    length: float
    length_diff: float


class GlueFinder:
    """Pairs the two copies of each number: one plain cut edge, one tab fold, same length."""

    def __init__(self, pieces: list[FlatPiece]):
        self.pieces = {p.id: p for p in pieces}

    def find(self) -> tuple[list[Glue], collections.Counter]:
        copies = collections.defaultdict(list)
        for piece in self.pieces.values():
            for label in piece.labels:
                if label.edge is not None and label.text.isdigit():
                    copies[label.text].append((piece, label))
        glues, status = [], collections.Counter()
        for number, found in copies.items():
            plain = [(p, self._oriented(p, l.edge)) for p, l in found if not l.tab]
            tabs = [(q, self._oriented(q, l.edge)) for q, l in found if l.tab]
            # A number read more than twice: the plain edge and tab that agree best are the pair (as edges.py).
            options = sorted((abs(p.length(*ab) - q.length(*cd)), p, ab, q, cd)
                             for p, ab in plain for q, cd in tabs if ab and cd)
            if not options:
                status["not one plain edge and one tab"] += 1
                continue
            diff, p, ab, q, cd = options[0]
            if diff > LENGTH_TOLERANCE_MM:
                status["length mismatch"] += 1
                continue
            status["glued" if len(found) == 2 else "glued, best of extra copies"] += 1
            glues.append(Glue(number, p.id, *ab, q.id, *cd, p.length(*ab), diff))
        return glues, status

    @staticmethod
    def _oriented(piece: FlatPiece, edge):
        key = frozenset(edge)
        for face in piece.faces:
            if face.tab:
                continue
            for a, b in piece.face_edges(face):
                if frozenset((a, b)) == key:
                    return a, b
        return None


# -- 3D assembly ------------------------------------------------------------------

class Assembly:
    """Every surface vertex of every piece as its own unknown 3D point. Pieces stay apart
    and glued edges are pulled together, so an edge that cannot close shows as a gap."""

    def __init__(self, pieces: list[FlatPiece]):
        self.pieces = {p.id: p for p in pieces}
        self.index: dict[tuple[str, int], int] = {}
        self.faces: list[tuple[str, list[int]]] = []       # piece, global vertex ids (ccw printed)
        self.apex: dict[tuple[int, int], int] = {}          # directed face edge -> another corner of that face
        for piece in pieces:
            for face in piece.faces:
                if face.tab:
                    continue
                verts = [self.index.setdefault((piece.id, v), len(self.index)) for v in face.vertices]
                self.faces.append((piece.id, verts))
                n = len(verts)
                for i in range(n):
                    self.apex[(verts[i], verts[(i + 1) % n])] = verts[(i + 2) % n]
        self.key = {g: k for k, g in self.index.items()}
        self.flat = np.array([self._flat(k) for k in self.index])

    def _flat(self, key) -> np.ndarray:
        piece, v = key
        x, y = self.pieces[piece].points[v]
        return np.array([x, -y])

    def vid(self, piece: str, v: int) -> int:
        return self.index[(piece, v)]

    def rigid_pairs(self) -> np.ndarray:
        """Every corner pair within a face at its flat distance: keeps each face flat and true."""
        seen = {}
        for _, verts in self.faces:
            for i, a in enumerate(verts):
                for b in verts[i + 1:]:
                    seen[(min(a, b), max(a, b))] = float(np.hypot(*(self.flat[a] - self.flat[b])))
        return np.array([(a, b, d) for (a, b), d in seen.items()])

    def folds(self) -> list[tuple[int, ...]]:
        """Folds inside pieces as hinges (a, b, p, a2, b2, q, sign): the left face runs a->b
        with corner p, the right face runs b2->a2 with corner q; sign +1 valley, -1 mountain."""
        rows = []
        for (a, b), p in self.apex.items():
            if a > b or (b, a) not in self.apex:
                continue
            piece, la = self.key[a]
            kind = self.pieces[piece].fold_kind.get(frozenset((la, self.key[b][1])))
            if kind:
                rows.append((a, b, p, a, b, self.apex[(b, a)], 1 if kind == "valley" else -1))
        return rows

    def seams(self, glues: list[Glue]) -> list[tuple[int, ...]]:
        """Glued edges as hinges, a meeting d and b meeting c. Pepakura draws a tab's fold in the
        style of the fold the glued edge makes in the model, so the tab gives the seam's direction."""
        rows = []
        for g in glues:
            a, b = self.vid(g.piece, g.a), self.vid(g.piece, g.b)
            c, d = self.vid(g.other, g.c), self.vid(g.other, g.d)
            kind = self.pieces[g.other].fold_kind.get(frozenset((g.c, g.d)))
            sign = {"valley": 1, "mountain": -1}.get(kind, 0)
            rows.append((a, b, self.apex[(a, b)], d, c, self.apex[(c, d)], sign))
        return rows

    def glue_pairs(self, glues: list[Glue]) -> np.ndarray:
        """(glue index, vertex, vertex) for both ends of every glued edge."""
        pairs = []
        for k, g in enumerate(glues):
            pairs.append((k, self.vid(g.piece, g.a), self.vid(g.other, g.d)))
            pairs.append((k, self.vid(g.piece, g.b), self.vid(g.other, g.c)))
        return np.array(pairs, dtype=np.int64).reshape(-1, 3)

    def open_edges(self, glues: list[Glue]) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
        """Plain cut edges and tab folds of the surface that no glue uses yet, in face order."""
        used = {(g.piece, g.a, g.b) for g in glues} | {(g.other, g.c, g.d) for g in glues}
        plain, tabs = [], []
        for pid, piece in self.pieces.items():
            sides = collections.defaultdict(list)
            for face in piece.faces:
                for a, b in piece.face_edges(face):
                    sides[frozenset((a, b))].append(face.tab)
            for face in piece.faces:
                if face.tab:
                    continue
                for a, b in piece.face_edges(face):
                    other = sides[frozenset((a, b))]
                    if (pid, a, b) in used:
                        continue
                    if len(other) == 1:
                        plain.append((pid, a, b))
                    elif True in other:
                        tabs.append((pid, a, b))
        return plain, tabs


# -- solver -----------------------------------------------------------------------

class FoldSolver:
    """Least squares over all vertex positions:

    * rigid: every pair of corners of a face keeps its flat distance (faces stay flat and true),
    * glue: both ends of a glued edge meet, with Cauchy weights redone each round so a wrong
      pair is let go and shows as a gap instead of crumpling the model around it,
    * fold: a valley fold or seam turns the faces towards the printed side, a mountain away,
    * smooth: a weak pull towards small bends, so parts the glue leaves loose rest in a
      gentle shell rather than crumpled.

    Terms can be limited to an `active` set of vertices (the pieces placed so far).
    """

    def __init__(self, assembly: Assembly, glues: list[Glue], fold_weight: float = 5.0,
                 smooth_weight: float = 1.0, gap_scale_mm: float = 2.0):
        self.asm = assembly
        self.glues = glues
        self.n = len(assembly.index)
        self.rigid = assembly.rigid_pairs()
        self.glue = assembly.glue_pairs(glues)
        folds = assembly.folds()
        self.hinge = np.array(folds + assembly.seams(glues), dtype=np.int64).reshape(-1, 7)
        self.hinge_glue = np.r_[np.full(len(folds), -1), np.arange(len(glues))]
        self.fold_weight = fold_weight
        self.smooth_weight = smooth_weight
        self.gap_scale = gap_scale_mm
        self.robust = np.ones(len(glues))      # Cauchy weight from the last solve
        self.dropped = np.zeros(len(glues), bool)
        rows = [(int(a), int(b)) for a, b, _ in self.rigid]
        rows += [(int(p), int(q)) for _, p, q in self.glue for _ in range(3)]
        rows += [tuple(int(v) for v in h[:6]) for h in self.hinge] * 2
        self.row_verts = rows

    @property
    def glue_weight(self) -> np.ndarray:
        return np.where(self.dropped, 0.0, self.robust)

    # residuals ------------------------------------------------------------------
    def _residual_selection(self, rows: np.ndarray) -> tuple[np.ndarray, ...]:
        """Split selected residual rows by term so local solves skip inactive geometry."""
        rigid_end = len(self.rigid)
        glue_end = rigid_end + 3 * len(self.glue)
        fold_end = glue_end + len(self.hinge)
        return (
            rows[rows < rigid_end],
            rows[(rows >= rigid_end) & (rows < glue_end)] - rigid_end,
            rows[(rows >= glue_end) & (rows < fold_end)] - glue_end,
            rows[rows >= fold_end] - fold_end,
        )

    def residuals(self, flat: np.ndarray, selection: tuple[np.ndarray, ...] | None = None) -> np.ndarray:
        x = flat.reshape(-1, 3)
        if selection is None:
            rigid_rows = np.arange(len(self.rigid))
            glue_rows = np.arange(3 * len(self.glue))
            fold_rows = smooth_rows = np.arange(len(self.hinge))
        else:
            rigid_rows, glue_rows, fold_rows, smooth_rows = selection

        parts = []
        if len(rigid_rows):
            a, b, d = self.rigid[rigid_rows].T
            parts.append(np.linalg.norm(x[a.astype(int)] - x[b.astype(int)], axis=1) - d)
        if len(glue_rows):
            pair, axis = np.divmod(glue_rows, 3)
            k, p, q = self.glue[pair].T
            parts.append(np.sqrt(self.glue_weight[k]) * (x[p, axis] - x[q, axis]))
        if len(fold_rows):
            hinge_glue = self.hinge_glue[fold_rows]
            weights = np.ones(len(fold_rows))
            glued = hinge_glue >= 0
            weights[glued] = self.glue_weight[hinge_glue[glued]]
            parts.append(self.fold_weight * weights * self._signs(x, fold_rows))
        if len(smooth_rows):
            hinge_glue = self.hinge_glue[smooth_rows]
            weights = np.ones(len(smooth_rows))
            glued = hinge_glue >= 0
            weights[glued] = self.glue_weight[hinge_glue[glued]]
            nl, nr = self._normals(x, smooth_rows)
            parts.append(self.smooth_weight * weights * np.linalg.norm(nl - nr, axis=1))
        return np.concatenate(parts) if parts else np.empty(0)

    def _normals(self, x: np.ndarray, indices: np.ndarray | None = None):
        hinge = self.hinge if indices is None else self.hinge[indices]
        a, b, p, a2, b2, q, _ = hinge.T
        nl = np.cross(x[b] - x[a], x[p] - x[a])
        nr = np.cross(x[a2] - x[b2], x[q] - x[b2])
        nl /= np.linalg.norm(nl, axis=1, keepdims=True) + 1e-9
        nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
        return nl, nr

    def _signs(self, x: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        """How far each fold bends the wrong way (0 when right or when its way is unknown)."""
        hinge = self.hinge if indices is None else self.hinge[indices]
        a, b, p, a2, b2, q, sign = hinge.T
        nl, _ = self._normals(x, indices)
        r = x[q] - x[a]
        s = np.einsum("ij,ij->i", nl, r) / (np.linalg.norm(r, axis=1) + 1e-9)
        return np.maximum(0.0, -sign * s)

    def bends(self, x: np.ndarray) -> np.ndarray:
        nl, nr = self._normals(x)
        return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nl, nr), -1, 1)))

    def solve(self, x: np.ndarray, free: list[int] | None = None, active: set[int] | None = None,
              rounds: int = 1, max_nfev: int = 100, reweight: bool = False) -> np.ndarray:
        """Move the `free` vertices (default all); use only terms whose vertices are all `active`.
        With `reweight`, each round after the first trusts a glue less the more it gapes."""
        from scipy.optimize import least_squares
        from scipy.sparse import csr_matrix

        free = np.arange(self.n) if free is None else np.asarray(sorted(free))
        col = np.full(self.n, -1)
        col[free] = np.arange(len(free))
        keep, ri, ci = [], [], []
        for r, verts in enumerate(self.row_verts):
            if active is not None and not all(v in active for v in verts):
                continue
            cols = [col[v] for v in verts if col[v] >= 0]
            if not cols:
                continue
            for c in cols:
                ri += [len(keep)] * 3
                ci += [3 * c, 3 * c + 1, 3 * c + 2]
            keep.append(r)
        keep = np.array(keep)
        pattern = csr_matrix((np.ones(len(ri), np.int8), (ri, ci)), shape=(len(keep), 3 * len(free)))
        selection = self._residual_selection(keep)
        flat = x.copy().ravel()
        idx = np.ravel(np.c_[3 * free, 3 * free + 1, 3 * free + 2])

        def fun(v):
            flat[idx] = v
            return self.residuals(flat, selection)
        for _ in range(rounds):
            flat[idx] = least_squares(fun, flat[idx].copy(), jac_sparsity=pattern, method="trf",
                                      x_scale="jac", max_nfev=max_nfev).x
            if reweight:
                self.robust = 1.0 / (1.0 + (self.gaps(flat.reshape(-1, 3)) / self.gap_scale) ** 2)
        return flat.reshape(-1, 3)

    # results --------------------------------------------------------------------
    def gaps(self, x: np.ndarray) -> np.ndarray:
        """Per glue: the larger distance between the two corner pairs that should meet."""
        out = np.zeros(len(self.glues))
        np.maximum.at(out, self.glue[:, 0], np.linalg.norm(x[self.glue[:, 1]] - x[self.glue[:, 2]], axis=1))
        return out

    def stretch(self, x: np.ndarray) -> np.ndarray:
        a, b, d = self.rigid[:, 0].astype(int), self.rigid[:, 1].astype(int), self.rigid[:, 2]
        return np.linalg.norm(x[a] - x[b], axis=1) - d

    def wrong_folds(self, x: np.ndarray) -> np.ndarray:
        open_or_unmapped = np.ones(len(self.hinge_glue), dtype=bool)
        glued = self.hinge_glue >= 0
        open_or_unmapped[glued] = ~self.dropped[self.hinge_glue[glued]]
        return (self._signs(x) > 0.05) & open_or_unmapped


# -- growing the model piece by piece ---------------------------------------------

def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proper rotation r and shift t with source @ r.T + t closest to target."""
    sc, tc = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - sc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, tc - sc @ r.T


class Grower:
    """Builds the model the way it is assembled: fold one piece along its printed folds,
    glue it onto the pieces already placed, settle the joint, repeat. Pieces join in order
    of how many glued edges tie them to what is built. A glue that still gapes after the
    joint settles is let go and reported: its number or one of its lines is likely wrong."""

    def __init__(self, solver: FoldSolver, bend_deg: float = 15.0, gape_mm: float = 3.0,
                 settle_every: int = 15, verbose: bool = False):
        self.s = solver
        self.asm = solver.asm
        self.bend = math.radians(bend_deg)
        self.gape = gape_mm
        self.settle_every = settle_every
        self.verbose = verbose
        self.faces_of = collections.defaultdict(list)
        for pid, verts in self.asm.faces:
            self.faces_of[pid].append(verts)
        self.verts_of = {pid: sorted({v for f in fs for v in f}) for pid, fs in self.faces_of.items()}
        self.links = collections.defaultdict(list)      # piece -> (glue index, other piece, ((own, other), ...))
        for k, g in enumerate(solver.glues):
            a, b = self.asm.vid(g.piece, g.a), self.asm.vid(g.piece, g.b)
            c, d = self.asm.vid(g.other, g.c), self.asm.vid(g.other, g.d)
            self.links[g.piece].append((k, g.other, ((a, d), (b, c))))
            self.links[g.other].append((k, g.piece, ((d, a), (c, b))))
        self.order: list[str] = []

    def folded(self, pid: str) -> np.ndarray:
        """The piece on its own, every inner fold bent `bend` the printed way. Rows follow verts_of[pid].
        Faces are laid out across the folds of a spanning tree; a corner shared by faces whose
        bends do not quite close around it takes the average, and the solver settles it."""
        verts = self.verts_of[pid]
        where = {v: i for i, v in enumerate(verts)}
        flat = np.c_[self.asm.flat[verts], np.zeros(len(verts))]
        faces = self.faces_of[pid]
        edge_face = {}
        for k, f in enumerate(faces):
            for i in range(len(f)):
                edge_face[(f[i], f[(i + 1) % len(f)])] = k
        kinds = {(int(h[0]), int(h[1])): int(h[6]) for h in self.s.hinge if h[0] == h[3] and h[1] == h[4]}
        pose = {0: (np.eye(3), np.zeros(3))}      # face -> rotation, shift taking flat to folded
        queue = [0]
        while queue:
            k = queue.pop(0)
            r, t = pose[k]
            f = faces[k]
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                other = edge_face.get((b, a))
                if other is None or other in pose:
                    continue
                sign = kinds.get((min(a, b), max(a, b)), 0)
                # +1 valley: the far face rises towards the printed (+z) side.
                pa, pb = flat[where[a]], flat[where[b]]
                hinge = _rotation(pb - pa, -sign * self.bend)
                pose[other] = (r @ hinge, r @ (pa - hinge @ pa) + t)
                queue.append(other)
        total, count = np.zeros_like(flat), np.zeros(len(verts))
        for k, (r, t) in pose.items():
            for v in faces[k]:
                total[where[v]] += flat[where[v]] @ r.T + t
                count[where[v]] += 1
        return total / np.maximum(count, 1)[:, None]

    def grow(self) -> np.ndarray:
        x = np.zeros((self.s.n, 3))
        pieces = list(self.verts_of)
        placed: set[str] = set()
        active: set[int] = set()
        while len(placed) < len(pieces):
            score = {p: self._ties(p, placed) for p in pieces if p not in placed}
            best = max(score, key=lambda p: (score[p], len(self.links[p])))
            self._place(best, x, placed)
            placed.add(best)
            active.update(self.verts_of[best])
            self.order.append(best)
            # Settle the new piece and its glued neighbours, then let go of glues that still gape.
            near = {best} | {o for _, o, _ in self.links[best] if o in placed}
            free = [v for p in near for v in self.verts_of[p]]
            x[:] = self.s.solve(x, free, active, max_nfev=30)
            self._let_go(best, x, placed)
            if len(self.order) % self.settle_every == 0:
                x[:] = self.s.solve(x, sorted(active), active, max_nfev=40)
                if self.verbose:
                    print(f"placed {len(self.order)} pieces, let go of {int(self.s.dropped.sum())} glues", flush=True)
        return x

    def _ties(self, pid: str, placed: set[str]) -> int:
        return sum(o in placed and not self.s.dropped[k] for k, o, _ in self.links[pid])

    def _let_go(self, pid: str, x: np.ndarray, placed: set[str]) -> None:
        gaps = self.s.gaps(x)
        mine = [k for k, o, _ in self.links[pid] if o in placed and not self.s.dropped[k]]
        while mine and max(gaps[k] for k in mine) > self.gape:
            worst = max(mine, key=lambda k: gaps[k])
            self.s.dropped[worst] = True
            mine.remove(worst)
            if mine:
                near = {pid} | {o for k, o, _ in self.links[pid] if o in placed}
                free = [v for p in near for v in self.verts_of[p]]
                active = {v for p in placed | {pid} for v in self.verts_of[p]}
                x[:] = self.s.solve(x, free, active, max_nfev=30)
                gaps = self.s.gaps(x)

    def _place(self, pid: str, x: np.ndarray, placed: set[str]) -> None:
        verts = self.verts_of[pid]
        local = self.folded(pid)
        where = {v: i for i, v in enumerate(verts)}
        links = [(k, ends) for k, o, ends in self.links[pid] if o in placed and not self.s.dropped[k]]
        pairs = [pair for _, ends in links for pair in ends]
        if not pairs:   # a new island: start it beside what is built
            built = [v for p in placed for v in self.verts_of[p]]
            shift = np.array([x[built, 0].max() + 40.0 if built else 0.0, 0, 0])
            x[verts] = local - local.mean(0) + shift
            return
        src = np.array([local[where[a]] for a, _ in pairs])
        dst = np.array([x[b] for _, b in pairs])
        if len(pairs) >= 3 and np.linalg.svd(src - src.mean(0), compute_uv=False)[1] > 1.0:
            r, t = _kabsch(src, dst)
            x[verts] = local @ r.T + t
            return
        # One glued edge: line it up, then swing the piece about it to bend the seam the printed way.
        u = (src[1] - src[0]) / np.linalg.norm(src[1] - src[0])
        v = (dst[1] - dst[0]) / np.linalg.norm(dst[1] - dst[0])
        r = _rotation(np.cross(u, v), math.acos(np.clip(u @ v, -1, 1))) if np.linalg.norm(np.cross(u, v)) > 1e-9 \
            else np.eye(3)
        base = (local - src[0]) @ r.T + dst[0]
        rows = [h for h, k in zip(self.s.hinge, self.s.hinge_glue) if k in {k for k, _ in links}]
        best, best_cost = base, np.inf
        for angle in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            x[verts] = _rotate(base, dst[0], v, angle)
            cost = self._seam_cost(x, rows)
            if cost < best_cost:
                best, best_cost = x[verts].copy(), cost
        x[verts] = best

    def _seam_cost(self, x: np.ndarray, rows) -> float:
        cost = 0.0
        for a, b, p, a2, b2, q, sign in rows:
            nl = np.cross(x[b] - x[a], x[p] - x[a])
            nr = np.cross(x[a2] - x[b2], x[q] - x[b2])
            nl /= np.linalg.norm(nl) + 1e-9
            nr /= np.linalg.norm(nr) + 1e-9
            r = x[q] - x[a]
            s = float(nl @ r) / (np.linalg.norm(r) + 1e-9)
            bend = math.acos(np.clip(nl @ nr, -1, 1))
            cost += 10 * max(0.0, -sign * s) + abs(bend - self.bend)
        return cost


def _rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points about the line through origin along axis."""
    return (points - origin) @ _rotation(axis, angle).T + origin


# -- proposing missing glue -------------------------------------------------------

class GlueProposer:
    """An open plain edge and an open tab fold of the same length whose ends already lie
    close together in the folded model are probably one glued edge whose numbers were
    misread or not paired. Proposals are tried in the model and kept only if they close."""

    def __init__(self, assembly: Assembly, length_mm: float = 0.6, reach: float = 0.35):
        self.asm = assembly
        self.length = length_mm
        self.reach = reach          # end distance allowed, as a share of the edge length

    def propose(self, x: np.ndarray, glues: list[Glue], skip: set[tuple] = frozenset()) -> list[Glue]:
        plain, tabs = self.asm.open_edges(glues)
        options = []
        for pid, a, b in plain:
            p = self.asm.pieces[pid]
            la = p.length(a, b)
            xa, xb = x[self.asm.vid(pid, a)], x[self.asm.vid(pid, b)]
            for qid, c, d in tabs:
                if qid == pid or (pid, a, b, qid, c, d) in skip:
                    continue
                q = self.asm.pieces[qid]
                diff = abs(la - q.length(c, d))
                if diff > self.length:
                    continue
                gap = max(np.linalg.norm(xa - x[self.asm.vid(qid, d)]), np.linalg.norm(xb - x[self.asm.vid(qid, c)]))
                if gap < self.reach * la:
                    options.append((gap, diff, pid, a, b, qid, c, d))
        found, used = [], set()
        for gap, diff, pid, a, b, qid, c, d in sorted(options):
            if (pid, a, b) in used or (qid, c, d) in used:
                continue
            used.update({(pid, a, b), (qid, c, d)})
            number = f"{self._read(pid, a, b) or '?'}/{self._read(qid, c, d) or '?'}"
            found.append(Glue(number, pid, a, b, qid, c, d, self.asm.pieces[pid].length(a, b), diff))
        return found

    def _read(self, pid: str, a: int, b: int) -> str:
        key = frozenset((a, b))
        return next((l.text for l in self.asm.pieces[pid].labels if l.edge and frozenset(l.edge) == key), "")


# -- the whole run ----------------------------------------------------------------

MATERIAL_RGB = {"red": (0.78, 0.16, 0.16), "green": (0.18, 0.49, 0.20), "yellow": (0.98, 0.66, 0.15),
                "black": (0.2, 0.2, 0.2), "white": (0.95, 0.95, 0.95), "beige": (0.93, 0.86, 0.70)}


# -- feeding the folded shape back to the flat pieces ------------------------------

class CornerAdjuster:
    """Where the folded model had to stretch a face to close its glues, the trace is likely a
    little off. Flatten each piece again from its folded shape: the 2D corners whose distances
    best match the folded ones, held lightly to where they were traced. A corner that moves a
    little becomes a `move` row (piece-edits.csv columns, page mm); one that moves a lot, or a
    piece with a numbered glue that does not close, is listed for a look by eye. A piece with a
    fold bent the wrong way sits in a pose the solver could not settle, so its stretch says more
    about the solver than the trace: it gets no moves."""

    MATCH_MM = 1.5    # as corrections.py: a move row takes the outline corner this close to its start

    def __init__(self, recon: "Reconstruction", min_mm: float = 0.2, max_mm: float = 1.0, hold: float = 0.05):
        self.recon = recon
        self.min_mm, self.max_mm, self.hold = min_mm, max_mm, hold

    def shifts(self) -> dict[str, dict[int, np.ndarray]]:
        """Per piece: outline corner id -> how far to move it (document mm, y down)."""
        from scipy.optimize import least_squares

        asm, x = self.recon.asm, self.recon.x
        closed = self._closed_vertices()
        by_piece = collections.defaultdict(list)
        for a, b, _ in asm.rigid_pairs():
            by_piece[asm.key[int(a)][0]].append((int(a), int(b)))
        out = {}
        for pid, piece in asm.pieces.items():
            verts = sorted({v for p, vs in asm.faces if p == pid for v in vs})
            if not verts or not any(v in closed for v in verts):
                continue          # nothing holds this piece's shape: no evidence either way
            local = {v: i for i, v in enumerate(verts)}
            pairs = np.array([(local[a], local[b], np.linalg.norm(x[a] - x[b])) for a, b in by_piece[pid]])
            start = np.array([asm.flat[v] for v in verts])
            i, j, d = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]

            def fun(f):
                q = f.reshape(-1, 2)
                return np.r_[np.linalg.norm(q[i] - q[j], axis=1) - d, self.hold * (q - start).ravel()]
            fitted = least_squares(fun, start.ravel()).x.reshape(-1, 2)
            # Drop any shift every corner shares (the piece as a whole sliding), keep the shape change.
            delta = fitted - start
            delta -= delta.mean(0)
            corners = {piece.vertex(c) for c in piece.outline}
            out[pid] = {asm.key[v][1]: np.array([dx, -dy]) for v, (dx, dy) in zip(verts, delta)
                        if asm.key[v][1] in corners}
        return out

    def _closed_vertices(self) -> set[int]:
        solver = self.recon.solver
        return {int(v) for k, p, q in solver.glue if not solver.dropped[k] for v in (p, q)}

    @staticmethod
    def carry(path: Path, moves: list[dict], match_mm: float = 0.05) -> list[dict]:
        """The build moves corners of the raw trace, so a run on an SVG that already has the
        last moves must keep them: a corner that now sits where an earlier row put it keeps that
        row, and moves on from the earlier row's start if it moves again. A run on the same SVG
        again replaces its own rows."""
        if not path.exists():
            return moves
        with path.open(newline="", encoding="utf-8") as handle:
            earlier = list(csv.DictReader(handle))
        point = lambda r, k: np.array([float(r[f"{k}x_mm"]), float(r[f"{k}y_mm"])])
        out, used = [], set()
        for row in moves:
            for i, old in enumerate(earlier):
                if old["piece"] == row["piece"] and min(np.hypot(*(point(old, k) - point(row, "a")))
                                                        for k in "ab") < match_mm:
                    row = {**row, "ax_mm": old["ax_mm"], "ay_mm": old["ay_mm"]}
                    used.add(i)
                    break
            out.append(row)
        return out + [old for i, old in enumerate(earlier) if i not in used]

    def unsettled(self) -> set[str]:
        solver = self.recon.solver
        return {self.recon.asm.key[int(h[0])][0] for h in solver.hinge[solver.wrong_folds(self.recon.x)]}

    def _movable(self, pid: str, v: int, shift: np.ndarray, worst: float) -> bool:
        piece = self.recon.asm.pieces[pid]
        if not self.min_mm <= np.hypot(*shift) <= self.max_mm or worst > self.max_mm:
            return False      # too small to matter, or a piece with one corner far off is looked at whole
        # the build would not know which of two near corners to move
        return sum(np.hypot(*(np.asarray(c) - piece.points[v])) < self.MATCH_MM for c in piece.outline) == 1

    def _hold_glue_lengths(self, chosen: dict[str, dict[int, np.ndarray]], slack_mm: float = 0.5) -> dict[str, list]:
        """Drop moves that would part a numbered pair: a glued edge and its tab that agree in
        length (within `slack_mm`, or as well as they did) must still agree once both pieces move."""
        held = collections.defaultdict(list)
        pieces, solver = self.recon.asm.pieces, self.recon.solver
        length = lambda pid, a, b: float(np.hypot(*(pieces[pid].points[a] + chosen.get(pid, {}).get(a, 0)
                                                     - pieces[pid].points[b] - chosen.get(pid, {}).get(b, 0))))
        for _ in range(3):
            changed = False
            for k, g in enumerate(solver.glues):
                if solver.dropped[k] or k in self.recon.proposed:
                    continue
                ends = [(g.piece, g.a), (g.piece, g.b), (g.other, g.c), (g.other, g.d)]
                if not any(v in chosen.get(pid, {}) for pid, v in ends):
                    continue
                before = abs(pieces[g.piece].length(g.a, g.b) - pieces[g.other].length(g.c, g.d))
                after = abs(length(g.piece, g.a, g.b) - length(g.other, g.c, g.d))
                if after > max(before, slack_mm):
                    for pid, v in ends:
                        if chosen.get(pid, {}).pop(v, None) is not None:
                            held[pid].append(g.number)
                            changed = True
            if not changed:
                break
        return held

    def rows(self) -> tuple[list[dict], list[dict]]:
        moves, inspect = [], []
        unsettled = self.unsettled()
        shifts = {pid: s for pid, s in self.shifts().items() if pid not in unsettled}
        open_numbers = collections.defaultdict(list)
        for r in self.recon.glue_rows():
            if r["status"] in ("does not close", "let go while building"):
                for pid in (r["piece"], r["tab_piece"]):
                    open_numbers[pid].append(f"{r['number']} ({r['gap_mm']} mm)")
        worst = {pid: max((float(np.hypot(*v)) for v in s.values()), default=0.0) for pid, s in shifts.items()}
        chosen = {pid: {v: shift for v, shift in s.items() if self._movable(pid, v, shift, worst[pid])}
                  for pid, s in shifts.items()}
        held = self._hold_glue_lengths(chosen)
        for pid in sorted(set(shifts) | set(open_numbers)):
            piece = self.recon.asm.pieces[pid]
            ox, oy = piece.page_origin
            for v, shift in sorted(chosen.get(pid, {}).items()):
                a = piece.points[v] - (ox, oy)
                b = a + shift
                moves.append({"piece": pid, "action": "move", "kind": "", "ax_mm": f"{a[0]:.2f}", "ay_mm": f"{a[1]:.2f}",
                              "bx_mm": f"{b[0]:.2f}", "by_mm": f"{b[1]:.2f}",
                              "note": f"3D fold: corner {np.hypot(*shift):.2f} mm off"})
            worst_mm = worst.get(pid, 0.0)
            reasons = []
            if worst_mm > self.max_mm:
                reasons.append(f"a corner {worst_mm:.1f} mm off, too far to move without a look")
            if held[pid]:
                reasons.append("moves held back, they would part glued edges: " + ", ".join(sorted(set(held[pid]))))
            if open_numbers[pid]:
                reasons.append("glue does not close: " + ", ".join(open_numbers[pid]))
            if reasons:
                inspect.append({"piece": pid, "worst_corner_mm": f"{worst_mm:.2f}" if pid in shifts else "",
                                "reason": "; ".join(reasons)})
        return moves, inspect


class Reconstruction:
    """Pattern SVG in; folded model, per-glue closure table and preview out."""

    def __init__(self, svg: Path, gape_mm: float = 3.0, propose_rounds: int = 3, verbose: bool = True,
                 reference: dict | None = None, anneal: int = 5, cold: bool = False):
        self.svg = svg
        self.state_path = svg.with_name("fold3d-state.npz")
        self.component_state_path = svg.with_name("fold3d-components.json")
        self.cold = cold
        self.reference = reference or {}   # finished model's counts (YAML `reference:`)
        self.gape = gape_mm
        self.rounds = propose_rounds
        self.verbose = verbose
        self.anneal = anneal               # cycles of hard then easy fold pull after the build

    def log(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def run(self) -> None:
        self.pieces = PatternReader().read(self.svg)
        overrides_path = self.svg.with_name("fold3d-tab-overrides.json")
        try:
            tab_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            tab_overrides = {}
        for piece in self.pieces:
            piece.split_faces()
            piece.mark_labels()
            piece.mark_tabs()
            edge_use = collections.Counter(frozenset(edge) for face in piece.faces for edge in piece.face_edges(face))
            for index, face in enumerate(piece.faces):
                override = tab_overrides.get(f"{piece.id}/{index}")
                if isinstance(override, bool):
                    face.tab = override
                    shared_edges = {frozenset(edge) for edge in piece.face_edges(face)
                                    if edge_use[frozenset(edge)] == 2}
                    labels = [label for label in piece.labels if label.edge and frozenset(label.edge) in shared_edges]
                    if labels:
                        centre = np.mean([piece.points[v] for v in face.vertices], axis=0)
                        label = min(labels, key=lambda item: np.linalg.norm(item.centre - centre))
                        label.tab = override
        self.asm = Assembly(self.pieces)
        glues, status = GlueFinder(self.pieces).find()
        self.log(f"{len(self.pieces)} pieces, {len(self.asm.faces)} faces; numbers: "
                 + ", ".join(f"{n} {s}" for s, n in status.most_common()))
        self.proposed: set[int] = set()
        solver = FoldSolver(self.asm, glues)
        grower = Grower(solver, gape_mm=self.gape, verbose=self.verbose)
        components, x = self._prebuild_components(solver, glues, grower)
        self.components = components
        x = self._anneal(solver, x, components)
        self._drop_gaping(solver, x)
        self.log(f"glued by number: {int((~solver.dropped).sum())} of {len(glues)} close")
        tried: set[tuple] = set()
        for _ in range(self.rounds):
            new = GlueProposer(self.asm).propose(x, solver.glues, tried)
            new = [g for g in new if (g.piece, g.a, g.b, g.other, g.c, g.d) not in tried]
            if not new:
                break
            tried.update((g.piece, g.a, g.b, g.other, g.c, g.d) for g in new)
            first = len(solver.glues)
            previous = {self._glue_key(g): solver.dropped[i] for i, g in enumerate(solver.glues)}
            solver = FoldSolver(self.asm, solver.glues + new)
            solver.dropped[:] = [previous.get(self._glue_key(g), False) for g in solver.glues]
            self.proposed.update(range(first, first + len(new)))
            x = self._join_components(solver, x, components, max_nfev=100)
            self._drop_gaping(solver, x)
            kept = sum(not solver.dropped[k] for k in range(first, first + len(new)))
            self.log(f"proposed {len(new)} glues by length and position, {kept} close")
        # Build again from scratch with the look-alike proposals as glue from the start: the
        # extra ties make the build stiffer, so fewer pieces settle in a wrong pose.
        keep = [k for k, g in enumerate(solver.glues)
                if k not in self.proposed or (not solver.dropped[k] and _look_alike(g.number) == "yes")]
        rest = [k for k in range(len(solver.glues)) if k not in keep]
        self.proposed = {i for i, k in enumerate(keep + rest) if k in self.proposed}
        # The other proposals stay listed, unglued, with their gap in the rebuilt model.
        glues = [solver.glues[k] for k in keep + rest]
        previous = {self._glue_key(g): solver.dropped[k] for k, g in enumerate(solver.glues)}
        solver = FoldSolver(self.asm, glues)
        solver.dropped[:] = [previous.get(self._glue_key(g), False) for g in solver.glues]
        self.grown_dropped = solver.dropped.copy()
        solver.dropped[len(keep):] = True
        solver.dropped[:len(keep)] = False
        solver.robust[:] = 1.0
        x = self._join_components(solver, x, components, max_nfev=160)
        x = self._anneal(solver, x, components)
        self._drop_gaping(solver, x)
        self.log(f"rebuilt with {len(keep)} glues, {int((~solver.dropped).sum())} close")
        self.solver, self.x = solver, x
        self._save_state(x)

    @staticmethod
    def _glue_key(glue: Glue) -> tuple:
        return glue.number, glue.piece, glue.a, glue.b, glue.other, glue.c, glue.d

    def _same_material_components(self, glues: list[Glue]) -> list[dict]:
        """Partition the piece graph by same-cardstock numbered glue connections."""
        piece_ids = sorted({piece for piece, _ in self.asm.index})
        parent = {piece: piece for piece in piece_ids}

        def find(piece):
            while parent[piece] != piece:
                parent[piece] = parent[parent[piece]]
                piece = parent[piece]
            return piece

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[max(a, b)] = min(a, b)

        for glue in glues:
            if glue.piece not in parent or glue.other not in parent:
                continue
            if self.asm.pieces[glue.piece].material == self.asm.pieces[glue.other].material:
                union(glue.piece, glue.other)

        members = collections.defaultdict(list)
        for piece in piece_ids:
            members[find(piece)].append(piece)
        components = []
        for pieces in sorted(members.values(), key=lambda row: (self.asm.pieces[row[0]].material, row)):
            vertices = sorted(v for v, (piece, _) in self.asm.key.items() if piece in pieces)
            components.append({"id": ",".join(pieces), "material": self.asm.pieces[pieces[0]].material,
                               "pieces": pieces, "vertices": vertices})
        return components

    def _component_signature(self, component: dict, glues: list[Glue]) -> str:
        selected = set(component["pieces"])
        pieces = []
        for piece_id in component["pieces"]:
            piece = self.asm.pieces[piece_id]
            pieces.append({
                "id": piece_id, "material": piece.material,
                "outline": piece.outline,
                "folds": piece.folds,
                "points": [point.tolist() for point in piece.points],
                "faces": [[face.vertices, face.tab] for face in piece.faces],
                "fold_kind": sorted((sorted(edge), kind) for edge, kind in piece.fold_kind.items()),
                "labels": [[label.text, label.edge, label.tab] for label in piece.labels],
            })
        internal = [self._glue_key(glue) for glue in glues
                    if glue.piece in selected and glue.other in selected]
        payload = {"solver": "same-cardstock-component-v1", "gape": self.gape,
                   "anneal": self.anneal, "fold_weight": 5.0, "smooth_weight": 1.0,
                   "pieces": pieces, "glues": sorted(internal)}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_component_cache(self) -> dict:
        if self.cold:
            return {}
        try:
            data = json.loads(self.component_state_path.read_text(encoding="utf-8"))
            return data.get("components", {}) if data.get("version") == 1 else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def _write_component_cache(self, cache: dict) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=self.component_state_path.parent,
                                             prefix="fold3d-components-tmp-", suffix=".json",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, "components": cache}, handle, separators=(",", ":"))
            temporary.replace(self.component_state_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _prebuild_components(self, solver: FoldSolver, glues: list[Glue], grower: Grower):
        components = self._same_material_components(glues)
        component_by_piece = {piece: i for i, component in enumerate(components)
                              for piece in component["pieces"]}
        cache = self._read_component_cache()
        signatures = {i: self._component_signature(component, glues)
                      for i, component in enumerate(components)}
        valid = {}
        for i, component in enumerate(components):
            record = cache.get(signatures[i])
            if not isinstance(record, dict) or not isinstance(record.get("vertices"), dict):
                continue
            keys = [self._vertex_key(vertex) for vertex in component["vertices"]]
            if all(key in record["vertices"] for key in keys):
                valid[i] = np.asarray([record["vertices"][key] for key in keys], dtype=float)

        warm = self._warm_start(grower)
        seed = warm
        seed_dropped = solver.dropped.copy()
        missing = [i for i in range(len(components)) if i not in valid]
        if seed is None and missing:
            # The cold path still uses the proven whole-model placer once to establish
            # useful component poses. Subsequent runs can skip it from the cache.
            seed = grower.grow()
            seed_dropped = solver.dropped.copy()
            self._save_state(seed)
            self.log("saved initial pose checkpoint; prebuilding cardstock components")
        elif seed is not None:
            self.log("reused the saved pose to align prebuilt cardstock components")

        x = np.zeros((solver.n, 3))
        for i, component in enumerate(components):
            vertices = component["vertices"]
            if i in valid:
                points = valid[i]
                if warm is not None and len(vertices) >= 3:
                    rotation, shift = _kabsch(points, warm[vertices])
                    points = points @ rotation.T + shift
                x[vertices] = points
                self.log(f"reused {component['material']} component {i + 1}/{len(components)} "
                         f"({len(component['pieces'])} pieces)")
                continue

            piece_set = set(component["pieces"])
            local_pieces = [piece for piece in self.pieces if piece.id in piece_set]
            local_asm = Assembly(local_pieces)
            local_global = [self.asm.vid(*local_asm.key[j]) for j in range(len(local_asm.key))]
            local_glue_global = [j for j, glue in enumerate(glues)
                                 if glue.piece in piece_set and glue.other in piece_set]
            local_glues = [glues[j] for j in local_glue_global]
            local_solver = FoldSolver(local_asm, local_glues)
            local_solver.dropped[:] = seed_dropped[local_glue_global]
            local_x = seed[local_global].copy()
            fixed = set(local_asm.faces[0][1][:3]) if local_asm.faces else set()
            free = [v for v in range(local_solver.n) if v not in fixed]
            if free:
                local_x = local_solver.solve(local_x, free=free, rounds=3, max_nfev=100, reweight=True)
                local_x = self._anneal_component(local_solver, local_x, free)
            local_solver.dropped |= local_solver.gaps(local_x) > self.gape
            for local_i, global_i in enumerate(local_global):
                x[global_i] = local_x[local_i]
            for local_i, global_i in enumerate(local_glue_global):
                solver.dropped[global_i] = local_solver.dropped[local_i]
            cache[signatures[i]] = {
                "id": component["id"], "material": component["material"],
                "vertices": {self._vertex_key(global_i): x[global_i].tolist()
                             for global_i in vertices},
            }
            self._write_component_cache(cache)
            self.log(f"prebuilt {component['material']} component {i + 1}/{len(components)} "
                     f"({len(component['pieces'])} pieces)")

        # Cross-component matches are solved as a smaller rigid-body problem below.
        # Retain their initial grow status for the glue report, but retry them in the join.
        self.grown_dropped = solver.dropped.copy()
        for k, glue in enumerate(glues):
            if component_by_piece.get(glue.piece) != component_by_piece.get(glue.other):
                self.grown_dropped[k] = bool(self.grown_dropped[k] or seed_dropped[k])
                solver.dropped[k] = False
        if not components:
            return components, seed if seed is not None else x
        return components, x

    def _anneal_component(self, solver: FoldSolver, x: np.ndarray, free: list[int]) -> np.ndarray:
        if not solver.wrong_folds(x).any():
            return x
        weight = solver.fold_weight
        for _ in range(self.anneal):
            for fold_weight, rounds in ((30 * weight, 1), (12 * weight, 1), (4 * weight, 2), (weight, 3)):
                solver.fold_weight = fold_weight
                solver.robust[:] = 1.0
                x = solver.solve(x, free=free, rounds=rounds, max_nfev=100, reweight=True)
        solver.fold_weight = weight
        return x

    def _join_components(self, solver: FoldSolver, x: np.ndarray, components: list[dict],
                         max_nfev: int) -> np.ndarray:
        """Close cross-component glues by moving cached subassemblies as rigid units."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        component_of_piece = {piece: i for i, component in enumerate(components)
                              for piece in component["pieces"]}
        vertex_component = np.full(solver.n, -1, dtype=int)
        for i, component in enumerate(components):
            vertex_component[component["vertices"]] = i
        cross_glue = [k for k, glue in enumerate(solver.glues)
                      if component_of_piece.get(glue.piece) != component_of_piece.get(glue.other)]
        cross_glue = [k for k in cross_glue if not solver.dropped[k]]
        if not cross_glue:
            return x
        rows = solver.glue[np.isin(solver.glue[:, 0], cross_glue)]
        rows = rows[vertex_component[rows[:, 1]] != vertex_component[rows[:, 2]]]
        if not len(rows):
            return x
        active_components = sorted(set(vertex_component[rows[:, 1]]) | set(vertex_component[rows[:, 2]]))
        links = collections.Counter()
        for k in cross_glue:
            glue = solver.glues[k]
            a, b = component_of_piece.get(glue.piece), component_of_piece.get(glue.other)
            if a is not None and b is not None:
                links[a] += 1; links[b] += 1
        anchor = max(active_components, key=lambda i: (links[i], len(components[i]["vertices"])))
        moving = [i for i in active_components if i != anchor]
        offsets = {component: 6 * j for j, component in enumerate(moving)}
        centres = {i: x[components[i]["vertices"]].mean(axis=0) for i in active_components}
        hinges = np.flatnonzero(np.isin(solver.hinge_glue, cross_glue))
        hglue = solver.hinge_glue[hinges]
        glue_ids, pa, pb = rows.T

        def place(parameters):
            posed = x.copy()
            for component in moving:
                start = offsets[component]
                rotation = Rotation.from_rotvec(parameters[start:start + 3])
                translation = parameters[start + 3:start + 6]
                vertices = components[component]["vertices"]
                posed[vertices] = rotation.apply(x[vertices] - centres[component]) + centres[component] + translation
            return posed

        def residual(parameters):
            posed = place(parameters)
            weights = np.sqrt(solver.glue_weight[glue_ids])
            parts = [(weights[:, None] * (posed[pa] - posed[pb])).ravel()]
            if len(hinges):
                on = np.where(hglue < 0, 1.0, solver.glue_weight[hglue])
                parts.append(solver.fold_weight * on * solver._signs(posed, hinges))
                nl, nr = solver._normals(posed, hinges)
                parts.append(solver.smooth_weight * on * np.linalg.norm(nl - nr, axis=1))
            return np.concatenate(parts)

        result = least_squares(residual, np.zeros(6 * len(moving)), method="trf",
                               x_scale="jac", max_nfev=max_nfev)
        return place(result.x)

    def _vertex_key(self, global_id: int) -> str:
        piece_id, local_id = self.asm.key[global_id]
        x, y = self.asm.pieces[piece_id].points[local_id]
        return f"{piece_id}:{x:.2f}:{y:.2f}"

    def _warm_start(self, grower: Grower) -> np.ndarray | None:
        """Reuse the prior solved pose for stable vertices; return None if it is too stale."""
        if self.cold:
            return None
        old_keys, old_x = [], None
        if self.state_path.is_file():
            try:
                with np.load(self.state_path, allow_pickle=False) as saved:
                    old_keys = json.loads(str(saved["keys"].item()))
                    old_x = np.asarray(saved["x"], dtype=float)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                old_x = None
        if old_x is None:
            # Older folds already contain the same correspondence in the viewer JSON.
            viewer = self.svg.with_name("fold3d.json")
            try:
                data = json.loads(viewer.read_text(encoding="utf-8"))
                old_x = np.asarray(data["vertices"], dtype=float)
                old_by_vertex = {}
                for face in data["faces"]:
                    for vertex, point in zip(face["v"], face["flat"]):
                        old_by_vertex[int(vertex)] = f"{face['piece']}:{point[0]:.2f}:{point[1]:.2f}"
                old_keys = [old_by_vertex[i] for i in range(len(old_x))]
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
                return None
        if old_x.ndim != 2 or old_x.shape[1] != 3 or len(old_keys) != len(old_x):
            return None
        old = dict(zip(old_keys, old_x))
        current_keys = [self._vertex_key(i) for i in range(grower.s.n)]
        matched = {i: old[key] for i, key in enumerate(current_keys) if key in old}
        if len(matched) < 0.7 * grower.s.n:
            self.log(f"saved 3D pose is stale ({len(matched)} of {grower.s.n} vertices match); building from scratch")
            return None

        x = np.zeros((grower.s.n, 3))
        for i, point in matched.items():
            x[i] = point
        for piece_id, verts in grower.verts_of.items():
            local = grower.folded(piece_id)
            known = [j for j, vertex in enumerate(verts) if vertex in matched]
            if not known:
                self.log(f"saved 3D pose cannot place new piece {piece_id}; building from scratch")
                return None
            source = local[known]
            target = np.array([x[verts[j]] for j in known])
            if len(known) >= 3:
                rotation, shift = _kabsch(source, target)
            elif len(known) == 2:
                a, b = source[1] - source[0], target[1] - target[0]
                if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
                    rotation, shift = np.eye(3), target[0] - source[0]
                else:
                    axis = np.cross(a, b)
                    if np.linalg.norm(axis) < 1e-9:
                        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
                        if np.linalg.norm(axis) < 1e-9:
                            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
                        angle = math.pi if a @ b < 0 else 0.0
                    else:
                        angle = math.acos(float(np.clip((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
                    rotation = _rotation(axis, angle)
                    shift = target[0] - source[0] @ rotation.T
            else:
                rotation, shift = np.eye(3), target[0] - source[0]
            placed = local @ rotation.T + shift
            for j, vertex in enumerate(verts):
                if vertex not in matched:
                    x[vertex] = placed[j]
        self.log(f"warm-started from {len(matched)} of {grower.s.n} unchanged vertices")
        return x

    def _save_state(self, x: np.ndarray | None = None) -> None:
        pose = self.x if x is None else x
        keys = [self._vertex_key(i) for i in range(len(pose))]
        # Replace atomically so an interrupted write never destroys the last good pose.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.state_path.parent, prefix="fold3d-state-tmp-",
                                             suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, keys=json.dumps(keys), x=pose)
            temporary.replace(self.state_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _anneal(self, solver: FoldSolver, x: np.ndarray, components: list[dict]) -> np.ndarray:
        """Anneal cross-component fold directions while preserving prebuilt component shapes."""
        weight = solver.fold_weight
        for cycle in range(self.anneal):
            for fold_weight, rounds in ((30 * weight, 1), (12 * weight, 1), (4 * weight, 2), (weight, 3)):
                solver.fold_weight = fold_weight
                x = self._join_components(solver, x, components, max_nfev=60 * rounds)
                solver.robust = 1.0 / (1.0 + (solver.gaps(x) / solver.gap_scale) ** 2)
            self.log(f"fold pull cycle {cycle + 1}: {int(solver.wrong_folds(x).sum())} folds the wrong way, "
                     f"{int((solver.gaps(x) <= self.gape).sum())} glues within {self.gape:g} mm")
        solver.fold_weight = weight
        return x

    def _drop_gaping(self, solver: FoldSolver, x: np.ndarray) -> None:
        solver.dropped |= solver.gaps(x) > self.gape

    # outputs --------------------------------------------------------------------
    def glue_rows(self) -> list[dict]:
        gaps = self.solver.gaps(self.x)
        rows = []
        for k, g in enumerate(self.solver.glues):
            proposed = k in self.proposed
            if proposed and _look_alike(g.number) != "yes":
                status = "proposed by shape only, not glued"
            elif self.solver.dropped[k]:
                status = "proposed, did not close" if proposed else (
                    "let go while building" if k < len(self.grown_dropped) and self.grown_dropped[k] else "does not close")
            else:
                status = "proposed, closes" if proposed else "closes"
            rows.append({"number": g.number, "piece": g.piece, "tab_piece": g.other, "length_mm": f"{g.length:.1f}",
                         "length_diff_mm": f"{g.length_diff:.2f}", "gap_mm": f"{gaps[k]:.1f}",
                         "look_alike": _look_alike(g.number) if proposed else "", "status": status})
        return sorted(rows, key=lambda r: (r["status"], -float(r["gap_mm"])))

    def write(self, out: Path, project: str) -> None:
        out.mkdir(parents=True, exist_ok=True)
        x = self._posed()
        self._obj(out / "fold3d.obj", x)
        rows = self.glue_rows()
        with (out / "fold3d-glue.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self._preview(out / "fold3d.png", x)
        self._viewer_json(out / "fold3d.json", x, rows)
        moves, inspect = CornerAdjuster(self).rows()
        moves = CornerAdjuster.carry(out / "fold3d-moves.csv", moves)
        self._csv(out / "fold3d-moves.csv", moves,
                  ["piece", "action", "kind", "ax_mm", "ay_mm", "bx_mm", "by_mm", "note"])
        self._csv(out / "fold3d-inspect.csv", inspect, ["piece", "worst_corner_mm", "reason"])
        self.log(f"corners to move: {len(moves)} in {len({m['piece'] for m in moves})} pieces; "
                 f"pieces to look at: {len(inspect)}; pieces with a fold bent the wrong way, left as traced: "
                 f"{len(CornerAdjuster(self).unsettled())}")
        if self.reference:
            self._baseline(out / "fold3d-baseline.csv")
        counts = collections.Counter(r["status"] for r in rows)
        wrong = int(self.solver.wrong_folds(self.x).sum())
        stretch = np.abs(self.solver.stretch(self.x))
        self.log("glues: " + ", ".join(f"{n} {s}" for s, n in counts.most_common())
                 + f"; folds bent the wrong way: {wrong}; faces stretched up to {stretch.max():.1f} mm")
        self._save_state()

    @staticmethod
    def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            writer.writerows(rows)

    def counts(self) -> dict[str, int]:
        """Triangles of the folded surface, and its corners once every closing glue joins its ends."""
        parent = list(range(self.solver.n))

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for k, p, q in self.solver.glue:
            if not self.solver.dropped[k]:
                parent[find(int(p))] = find(int(q))
        corners = len({find(v) for _, verts in self.asm.faces for v in verts})
        return {"triangles": sum(len(verts) - 2 for _, verts in self.asm.faces), "corners": corners}

    def _baseline(self, path: Path) -> None:
        """The folded trace against the finished model's counts from the project YAML."""
        counts = self.counts()
        rows = [{"measure": key, "traced": counts[key], "reference": int(self.reference[key]),
                 "difference": counts[key] - int(self.reference[key])}
                for key in ("triangles", "corners") if key in self.reference]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, ["measure", "traced", "reference", "difference"])
            writer.writeheader()
            writer.writerows(rows)
        self.log("against the finished model: " + ", ".join(
            f"{r['measure']} {r['traced']} vs {r['reference']} ({r['difference']:+d})" for r in rows))

    def _posed(self) -> np.ndarray:
        """Centred on the largest glued part, longest axis up."""
        x = self.x - self.x.mean(0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        x = x @ vt.T
        return x[:, [1, 2, 0]]

    def _viewer_json(self, path: Path, x: np.ndarray, rows: list[dict]) -> None:
        """Everything app/fold3d-viewer.html draws, with ids that stay put between the 3D model and
        the pattern: face `piece-PP-NN/k` is the k-th surface face of that piece group in the SVG,
        with its corners both folded (vertex ids) and flat (document mm, as the SVG draws them)."""
        import json

        faces, seen = [], collections.Counter()
        for pid, verts in self.asm.faces:
            faces.append({"id": f"{pid}/{seen[pid]}", "piece": pid, "tab": self.asm.pieces[pid].faces[seen[pid]].tab, "v": verts,
                          "flat": [[round(float(c), 2) for c in self.asm.pieces[pid].points[self.asm.key[v][1]]]
                                   for v in verts]})
            seen[pid] += 1
        status = {(r["number"], r["piece"], r["tab_piece"]): r for r in rows}
        gaps = self.solver.gaps(self.x)
        glues = []
        for k, g in enumerate(self.solver.glues):
            r = status[(g.number, g.piece, g.other)]
            try:
                ends = [self.asm.vid(g.piece, g.a), self.asm.vid(g.piece, g.b),
                        self.asm.vid(g.other, g.c), self.asm.vid(g.other, g.d)]
            except KeyError:
                continue
            glues.append({"number": g.number, "piece": g.piece, "tab_piece": g.other, "v": ends,
                          "gap_mm": round(float(gaps[k]), 1), "status": r["status"]})
        wrong = self.solver.wrong_folds(self.x)
        inspect = {row["piece"]: row["reason"] for row in CornerAdjuster(self).rows()[1]}
        data = {"source": self.svg.name, "units": "mm",
                "vertices": [[round(float(c), 2) for c in p] for p in x],
                "pieces": {pid: {"material": piece.material, "inspect": inspect.get(pid, "")}
                           for pid, piece in self.asm.pieces.items()},
                "faces": faces, "glues": glues,
                "wrong_folds": [[int(h[0]), int(h[1])] for h in self.solver.hinge[wrong]],
                "colours": {m: "#%02x%02x%02x" % tuple(int(255 * c) for c in rgb) for m, rgb in MATERIAL_RGB.items()}}
        path.write_text(json.dumps(data, separators=(",", ":")))

    def _obj(self, path: Path, x: np.ndarray) -> None:
        mtl = path.with_suffix(".mtl")
        materials = sorted({p.material for p in self.pieces})
        mtl.write_text("".join(f"newmtl {m}\nKd {' '.join(f'{c:.3f}' for c in MATERIAL_RGB.get(m, (0.7, 0.7, 0.7)))}\n\n"
                               for m in materials))
        lines = [f"# folded from {self.svg.name}; millimetres", f"mtllib {mtl.name}"]
        lines += [f"v {a:.3f} {b:.3f} {c:.3f}" for a, b, c in x]
        by_piece = collections.defaultdict(list)
        for pid, verts in self.asm.faces:
            by_piece[pid].append(verts)
        for pid, faces in by_piece.items():
            lines += [f"o {pid}", f"usemtl {self.asm.pieces[pid].material}"]
            lines += ["f " + " ".join(str(v + 1) for v in verts) for verts in faces]
        path.write_text("\n".join(lines) + "\n")

    def _preview(self, path: Path, x: np.ndarray) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
        polys, colours = [], []
        for pid, verts in self.asm.faces:
            p = x[verts]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            shade = 0.45 + 0.55 * abs(float(n @ light)) / (np.linalg.norm(n) + 1e-9)
            polys.append(p)
            colours.append(tuple(c * shade for c in MATERIAL_RGB.get(self.asm.pieces[pid].material, (0.7, 0.7, 0.7))))
        fig = plt.figure(figsize=(15, 5.4))
        m = np.abs(x).max()
        for i, (elev, azim, title) in enumerate([(10, -90, "front"), (10, 0, "side"), (80, -90, "top")]):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors=(0, 0, 0, 0.25), linewidths=0.2))
            ax.set_xlim(-m, m), ax.set_ylim(-m, m), ax.set_zlim(-m, m)
            ax.view_init(elev, azim)
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def _look_alike(number: str) -> str:
    """For a proposed glue 'a/b': yes when the two reads differ in one digit or one dropped digit."""
    a, _, b = number.partition("/")
    if "?" in (a, b):
        return ""
    if len(a) == len(b):
        return "yes" if sum(x != y for x, y in zip(a, b)) == 1 else "no"
    short, long = sorted((a, b), key=len)
    return "yes" if len(long) - len(short) == 1 and any(long[:i] + long[i + 1:] == short for i in range(len(long))) else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold the traced pattern back into the 3D model.")
    parser.add_argument("--config", type=Path, default=Path("data/example/example.yaml"), help="project YAML manifest")
    parser.add_argument("--gape", type=float, default=3.0, help="let go of a glue whose ends stay this far apart (mm)")
    parser.add_argument("--anneal", type=int, default=5, help="cycles of hard then easy fold pull after the build")
    parser.add_argument("--cold", action="store_true", help="ignore saved 3D poses and place pieces from scratch")
    args = parser.parse_args()
    project = args.config.stem
    out = args.config.parent
    svg = out / f"{project}.svg"
    if not svg.is_file():
        parser.error(f"{svg} not found; run make first")
    import yaml
    reference = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("reference") or {}
    run = Reconstruction(svg, gape_mm=args.gape, reference=reference, anneal=args.anneal, cold=args.cold)
    run.run()
    run.write(out, project)
    print(f"Wrote {out / 'fold3d.obj'}, fold3d-glue.csv, fold3d-baseline.csv, fold3d-moves.csv, fold3d-inspect.csv and fold3d.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
