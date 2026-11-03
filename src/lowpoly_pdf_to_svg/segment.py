"""Cut a folded model into pieces from scratch, from its 3D mesh (lowpoly #68).

The PDF's split into pieces is one answer among many. Here the faces of fold3d.json are
welded into one mesh (corners within 1 mm are one vertex) and every mesh edge is decided
again: a fold inside a piece, or a seam between pieces. A seam needs a tab and glue, and a
seam shorter than one tape strip (15 mm) barely holds, so the search turns short seams into
folds while every piece still lies flat without overlapping itself and fits the sheet.

* `Mesh`: the welded faces, which faces share an edge, and the mirror map (x -> -x).
* `Layout`: one piece laid flat, face by face along the folds it was merged across.
* `Segmentation`: pieces as merged faces, their seams, and the score.
* `Segmenter`: greedy best-first merging; each merge is applied to both mirror twins and
  kept only if the merged pieces still lie flat and fit the sheet.

Only colour boundaries are fixed: faces of different cardstock never merge. The PDF's
pieces are rebuilt the same way (`Segmentation.from_pieces`) as the baseline to compare.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from .stress import SeamLoad

CLUSTER_MM = 1.0      # 3D corners this close are one vertex
TAPE_MM = 15.0        # a seam shorter than one strip of tape is weak
HINGE_MM = 0.5        # two faces' copies of an edge this close in the layout: a fold
OVERLAP_MM2 = 2.0     # overlap allowed in a layout (rounding between faces' flat shapes)
STRAIGHT_DEG = 15.0  # a seam bending more than this starts a new run: a new tab and tape strip
MIRROR_MM = 10.0      # a face's mirror: the face nearest its reflection, both ways, within this
DART_MM = 0.0         # a dart's slit narrower than this counts as a short seam (partition.py --dart-mm)
TINY_MM2 = 600.0     # a piece smaller than this (about 25 x 25 mm) is fiddly to fold and glue
SHEET_MM = (180.0, 249.0)   # A4 and Letter (the smaller of each side) less a 15 mm border for tabs
SHEET_AREA = SHEET_MM[0] * SHEET_MM[1]
MAX_PIECE_MM = SHEET_MM       # largest piece allowed (partition.py --max-piece): smaller pieces pack better
BOX_WEIGHT = 20 / SHEET_AREA  # a sheet's worth of bounding boxes costs as much as one piece


@dataclass
class Face:
    id: str
    piece: str            # the PDF's piece
    material: str
    corners: list[int]    # welded vertices, in the order of `flat`
    flat: np.ndarray      # the face's own paper shape, mm

    def edges(self):
        n = len(self.corners)
        for k in range(n):
            yield k, self.corners[k], self.corners[(k + 1) % n]


class Mesh:
    """The model's faces (no tabs) welded into one mesh."""

    def __init__(self, data: dict):
        self.data = data
        raw = np.array(data["vertices"], float)
        weld = self._weld(raw)
        ids = sorted(set(weld))
        index = {w: i for i, w in enumerate(ids)}
        groups = defaultdict(list)
        for i, w in enumerate(weld):
            groups[index[w]].append(i)
        self.points = np.array([raw[groups[i]].mean(axis=0) for i in range(len(ids))])
        self.faces: list[Face] = []
        for f in data["faces"]:
            if f["tab"]:
                continue
            corners, flat = [], []
            for v, xy in zip(f["v"], f["flat"]):
                w = index[weld[v]]
                if not corners or corners[-1] != w:
                    corners.append(w)
                    flat.append(xy)
            if len(corners) > 1 and corners[0] == corners[-1]:
                corners.pop(), flat.pop()
            if len(corners) >= 3:
                material = data["pieces"].get(f["piece"], {}).get("material", "")
                self.faces.append(Face(f["id"], f["piece"], material, corners, np.array(flat, float)))
        self.edge_faces: dict[frozenset, list[tuple[int, int]]] = defaultdict(list)
        for i, face in enumerate(self.faces):
            for k, a, b in face.edges():
                self.edge_faces[frozenset((a, b))].append((i, k))
        self.mirror_vertex, self.mirror_face = self._mirror()
        self.moved_mm = 0.0

    def symmetrize(self) -> float:
        """Make the mesh exactly mirror-symmetric and take each face's paper shape from it.

        Each vertex goes to the average of itself and its twin's reflection (centre-line
        vertices onto the centre plane), so left and right pieces lay out alike and an
        overlap test cannot pass on one side and fail on the other. Returns the largest
        move, mm: a small geometry edit, as Helga allowed (lowpoly #68)."""
        new = self.points.copy()
        for i, j in self.mirror_vertex.items():
            new[i] = (self.points[i] + self.flip(self.points[j])) / 2
        moved = float(np.max(np.linalg.norm(new - self.points, axis=1)))
        self.points = new
        for f in self.faces:
            f.flat = _flatten(self.points[f.corners], f.flat)
        self.moved_mm = moved
        return moved

    @staticmethod
    def _weld(points: np.ndarray) -> list[int]:
        parent = list(range(len(points)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in cKDTree(points).query_pairs(CLUSTER_MM):
            parent[find(i)] = find(j)
        return [find(i) for i in range(len(points))]

    def defect(self, v: int) -> float | None:
        """Angle defect at vertex v, degrees (360 less the face angles round it); None on
        the rim. Near 0 the paper round v is almost flat, and a dart there is a thin slit."""
        total = 0.0
        for i, f in enumerate(self.faces):
            if v in f.corners:
                k, n = f.corners.index(v), len(f.corners)
                a = self.points[f.corners[k - 1]] - self.points[v]
                b = self.points[f.corners[(k + 1) % n]] - self.points[v]
                total += math.degrees(math.acos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
        return 360.0 - total

    def interior(self, v: int) -> bool:
        edges = [e for e in self.edge_faces if v in e]
        return bool(edges) and all(len(self.edge_faces[e]) == 2 for e in edges)

    def flatten_vertices(self, faces: set[int], max_deg: float = 6.0, max_mm: float = 3.0) -> list[dict]:
        """Geometry edit: push each nearly flat vertex of these faces (defect under
        `max_deg`) along its normal until the paper round it is flat, if that takes no more
        than `max_mm`; its mirror twin moves the same way. A flat vertex needs no dart, so
        the pieces round it fold without a slit. Returns one row per vertex moved."""
        verts = sorted({v for i in faces for v in self.faces[i].corners})
        done, rows = set(), []
        for v in verts:
            if v in done or not self.interior(v):
                continue
            d0 = self.defect(v)
            if abs(d0) < 0.2 or abs(d0) > max_deg:
                continue
            ring = [i for i, f in enumerate(self.faces) if v in f.corners]
            normal = sum(np.cross(*(self.points[self.faces[i].corners[1:3]] - self.points[self.faces[i].corners[0]]))
                         for i in ring)
            normal /= np.linalg.norm(normal) or 1.0
            start = self.points[v].copy()

            def at(t):
                self.points[v] = start + t * normal
                return self.defect(v)
            lo, hi = -max_mm, max_mm
            f_lo, f_hi = at(lo), at(hi)
            if f_lo * f_hi > 0:
                self.points[v] = start
                continue
            for _ in range(40):
                mid = (lo + hi) / 2
                f_mid = at(mid)
                if f_lo * f_mid <= 0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid
            t = (lo + hi) / 2
            at(t)
            m = self.mirror_vertex.get(v)
            if m is not None and m != v:
                self.points[m] = self.flip(self.points[v])
                done.add(m)
            elif m == v:
                self.points[v][0] = self.mid      # on the centre line: stay on it
            done.add(v)
            rows.append({"vertex": v, "defect_deg": round(d0, 2), "moved_mm": round(abs(t), 2)})
        for f in self.faces:
            f.flat = _flatten(self.points[f.corners], f.flat)
        return rows

    def centre(self, i: int) -> np.ndarray:
        return self.points[self.faces[i].corners].mean(axis=0)

    def _mirror(self):
        self.mid = float(np.median(self.points[:, 0]))
        flip = self.flip
        def mutual(points):    # i <-> j: each is the other's nearest reflection
            d, j = cKDTree(points).query(flip(points))
            return {i: int(j[i]) for i in range(len(points)) if d[i] <= MIRROR_MM and j[j[i]] == i}
        centres = np.array([self.centre(i) for i in range(len(self.faces))])
        return mutual(self.points), mutual(centres)

    def flip(self, p: np.ndarray) -> np.ndarray:
        return p * [-1, 1, 1] + [2 * self.mid, 0, 0]

    def length(self, edge: frozenset) -> float:
        a, b = tuple(edge)
        return float(np.linalg.norm(self.points[a] - self.points[b]))

    def mirror_edge(self, edge: frozenset, i: int, j: int) -> frozenset | None:
        """The edge between faces i and j nearest the reflection of `edge`."""
        shared = [e for e, fs in self.edge_faces.items() if {f for f, _ in fs} >= {i, j}] \
            if i != j else []
        if not shared:
            return None
        target = self.flip(self.points[list(edge)].mean(axis=0))
        return min(shared, key=lambda e: np.linalg.norm(self.points[list(e)].mean(axis=0) - target))


def _flatten(points: np.ndarray, like: np.ndarray) -> np.ndarray:
    """A face's 3D corners laid in its best-fit plane, turned the same way up as `like`."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    flat = (points - centre) @ vt[:2].T
    if np.sign(_shoelace(flat)) != np.sign(_shoelace(like)):
        flat[:, 1] *= -1
    return flat


def _rigid(src: np.ndarray, dst: np.ndarray):
    """Rotation + translation taking segment src onto segment dst (no reflection)."""
    a = math.atan2(*(dst[1] - dst[0])[::-1]) - math.atan2(*(src[1] - src[0])[::-1])
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = dst.mean(axis=0) - src.mean(axis=0) @ r.T
    return lambda p: p @ r.T + t


def _shoelace(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def segment_box_weight() -> float:
    return BOX_WEIGHT


class Layout:
    """One piece laid flat: each face's corners in the piece's paper coordinates."""

    def __init__(self, mesh: Mesh, placed: dict[int, np.ndarray]):
        self.mesh, self.placed = mesh, placed

    @classmethod
    def single(cls, mesh: Mesh, i: int) -> "Layout":
        return cls(mesh, {i: mesh.faces[i].flat.copy()})

    def corner(self, i: int, vertex: int) -> np.ndarray:
        return self.placed[i][self.mesh.faces[i].corners.index(vertex)]

    def polygons(self) -> list[Polygon]:
        return [Polygon(p).buffer(0) for p in self.placed.values()]

    def outline(self):
        return unary_union([p.buffer(0.05) for p in self.polygons()]).buffer(-0.05)

    def joined(self, other: "Layout", edge: frozenset, mine: int, theirs: int) -> "Layout | None":
        """This piece and `other` hinged on `edge` (face `mine` here, `theirs` there), or None
        if the two overlap."""
        a, b = tuple(edge)
        move = _rigid(np.array([other.corner(theirs, a), other.corner(theirs, b)]),
                      np.array([self.corner(mine, a), self.corner(mine, b)]))
        moved = {i: move(p) for i, p in other.placed.items()}
        here = unary_union(self.polygons())
        there = unary_union([Polygon(p).buffer(0) for p in moved.values()])
        if here.intersection(there).area > OVERLAP_MM2:
            return None
        return Layout(self.mesh, {**self.placed, **moved})

    def fits(self, sheet=None) -> bool:
        sheet = sheet or MAX_PIECE_MM
        sides = self.size()
        return sides[0] <= min(sheet) and sides[1] <= max(sheet)

    def size(self) -> list[float]:
        """Short and long side of the smallest rectangle round the piece, mm."""
        box = MultiPoint(np.vstack(list(self.placed.values()))).minimum_rotated_rectangle
        xy = np.array(box.exterior.coords)[:4]
        return sorted(np.linalg.norm(xy[1:3] - xy[0:2], axis=1))

    def box_area(self) -> float:
        """Area of the smallest rectangle round the piece: what it takes on a sheet, roughly.
        Long, bent or spiky pieces waste card between them."""
        a, b = self.size()
        return a * b

    def area(self) -> float:
        return sum(abs(_shoelace(p)) for p in self.placed.values())

    def gap(self, edge: frozenset, i: int, j: int) -> float:
        """How far apart the two copies of an edge lie in the layout: 0 on a fold, the
        width of the slit's mouth on a dart."""
        return max(float(np.linalg.norm(self.corner(i, v) - self.corner(j, v))) for v in edge)

    def is_fold(self, edge: frozenset, i: int, j: int) -> bool:
        return self.gap(edge, i, j) <= HINGE_MM


class Segmentation:
    """Every face's piece, each piece's layout, and the seams between them."""

    def __init__(self, mesh: Mesh, region: set[int]):
        self.mesh, self.region = mesh, region
        self.piece = {i: f"f{i}" for i in range(len(mesh.faces))}
        self.layout = {f"f{i}": Layout.single(mesh, i) for i in region}
        self.pool: dict[frozenset, Layout] | None = None    # every piece merged, for partition.py
        for i in set(range(len(mesh.faces))) - region:   # outside the region: the PDF's pieces
            self.piece[i] = mesh.faces[i].piece

    def copy(self) -> "Segmentation":
        new = Segmentation.__new__(Segmentation)
        new.mesh, new.region = self.mesh, self.region
        new.piece, new.layout = dict(self.piece), dict(self.layout)
        new.pool = self.pool
        if hasattr(self, "_edges"):
            new._edges = self._edges
        return new

    @classmethod
    def from_pieces(cls, mesh: Mesh, region: set[int]) -> "Segmentation":
        """The PDF's pieces, rebuilt by merging each one's faces across their shared edges."""
        seg = cls(mesh, region)
        changed = True
        while changed:
            changed = False
            for edge, faces in mesh.edge_faces.items():
                if len(faces) != 2:
                    continue
                (i, _), (j, _) = faces
                if i in region and j in region and mesh.faces[i].piece == mesh.faces[j].piece \
                        and seg.piece[i] != seg.piece[j] and seg.merge(edge, i, j, check=False):
                    changed = True
        return seg

    def members(self, label: str) -> list[int]:
        return sorted(self.layout[label].placed)

    def merge(self, edge: frozenset, i: int, j: int, check: bool = True) -> bool:
        p, q = self.piece[i], self.piece[j]
        if p == q or p not in self.layout or q not in self.layout:
            return False
        joined = self.layout[p].joined(self.layout[q], edge, i, j)
        if joined is None or (check and not joined.fits()):
            return False
        for f in joined.placed:
            self.piece[f] = p
        self.layout[p] = joined
        del self.layout[q]
        if self.pool is not None:
            self.pool.setdefault(frozenset(joined.placed), joined)
        return True

    @classmethod
    def from_layouts(cls, mesh: Mesh, region: set[int], layouts: list[Layout]) -> "Segmentation":
        """The pieces given (they must cover the region exactly once)."""
        seg = cls(mesh, region)
        seg.layout = {}
        for lay in layouts:
            label = f"f{min(lay.placed)}"
            seg.layout[label] = lay
            for f in lay.placed:
                seg.piece[f] = label
        return seg

    def edge_kind(self, edge: frozenset) -> str:
        faces = self.mesh.edge_faces[edge]
        if len(faces) == 1:
            return "rim"
        (i, _), (j, _) = faces[:2]
        p, q = self.piece[i], self.piece[j]
        if p != q:
            return "seam"
        if p in self.layout:
            return "fold" if self.layout[p].is_fold(edge, i, j) else "dart"   # dart: glued to itself
        return "fold"

    def seams(self) -> list[dict]:
        """Seams touching the region, split into runs of edges that meet end to end."""
        by_pair = defaultdict(list)
        if not hasattr(self, "_edges"):
            self._edges = [(e, fs) for e, fs in self.mesh.edge_faces.items()
                           if len(fs) >= 2 and any(f in self.region for f, _ in fs)]
        for edge, faces in self._edges:
            kind = self.edge_kind(edge)
            if kind in ("seam", "dart"):
                key = tuple(sorted({self.piece[faces[0][0]], self.piece[faces[1][0]]}))
                by_pair[key].append(edge)
        runs = []
        for key, edges in by_pair.items():
            left = set(edges)
            while left:
                run, ends = [left.pop()], None
                ends = set(run[0])
                grew = True
                while grew:
                    grew = False
                    for e in list(left):
                        if e & ends and self._straight(run, e):
                            run.append(e)
                            left.remove(e)
                            ends ^= e
                            grew = True
                runs.append({"pieces": key, "edges": run,
                             "length_mm": sum(self.mesh.length(e) for e in run)})
        return runs

    def _thin(self, run: dict) -> bool:
        """A dart whose slit is narrower than a tab: its glue flap is too thin to hold."""
        if len(run["pieces"]) != 1:
            return False
        lay = self.layout[run["pieces"][0]]
        return max(lay.gap(e, *[f for f, _ in self.mesh.edge_faces[e][:2]]) for e in run["edges"]) < DART_MM

    def _straight(self, run: list[frozenset], e: frozenset) -> bool:
        """`e` continues the run in a straight line (one tab, one tape strip)."""
        for r in run:
            if r & e:
                (shared,) = r & e
                (a,), (b,) = r - {shared}, e - {shared}
                u = self.mesh.points[shared] - self.mesh.points[a]
                v = self.mesh.points[b] - self.mesh.points[shared]
                return u @ v >= math.cos(math.radians(STRAIGHT_DEG)) * np.linalg.norm(u) * np.linalg.norm(v)
        return False

    def score(self) -> tuple[float, dict]:
        runs = self.seams()
        short = sum(r["length_mm"] < TAPE_MM or self._thin(r) for r in runs)
        darts = sum(len(r["pieces"]) == 1 for r in runs)
        labels = {self.piece[i] for i in self.region}
        tiny = sum(self.layout[lab].area() < TINY_MM2 for lab in labels)
        length = sum(r["length_mm"] for r in runs)
        box = sum(self.layout[lab].box_area() for lab in labels)
        parts = {"short_seams": short, "seams": len(runs), "darts": darts, "pieces": len(labels),
                 "tiny_pieces": tiny, "seam_mm": round(length, 1), "sheets": round(box / SHEET_AREA, 2)}
        return (100 * short + 60 * tiny + 20 * len(labels) + 5 * len(runs) + 10 * darts + length / 10
                + segment_box_weight() * box), parts

    def mirror_label(self, label: str) -> str | None:
        m = self.mesh.mirror_face.get(self.members(label)[0])
        return self.piece[m] if m is not None else None

    def fold3d(self) -> dict:
        """The model with these pieces, in fold3d.json's shape, for SeamLoad."""
        faces = [{"piece": self.piece[i], "v": f.corners} for i, f in enumerate(self.mesh.faces)]
        glues = []
        for edge, fs in self.mesh.edge_faces.items():
            if len(fs) >= 2 and self.piece[fs[0][0]] != self.piece[fs[1][0]]:
                a, b = tuple(edge)
                glues.append({"piece": self.piece[fs[0][0]], "tab_piece": self.piece[fs[1][0]],
                              "v": [a, b, a, b], "status": "closes"})
        return {"vertices": self.mesh.points.tolist(), "faces": faces, "glues": glues}


class Segmenter:
    """Greedy best-first merging inside the region, mirror twins together."""

    def __init__(self, mesh: Mesh, region: set[int]):
        self.mesh, self.region = mesh, region

    def candidates(self, seg: Segmentation):
        pairs = defaultdict(list)
        for edge, faces in self.mesh.edge_faces.items():
            if len(faces) != 2:
                continue
            (i, _), (j, _) = faces
            if i in self.region and j in self.region and seg.piece[i] != seg.piece[j] \
                    and self.mesh.faces[i].material == self.mesh.faces[j].material:
                pairs[frozenset((seg.piece[i], seg.piece[j]))].append((edge, i, j))
        for options in pairs.values():
            yield sorted(options, key=lambda o: -self.mesh.length(o[0]))

    def _mirrored(self, seg: Segmentation, edge, i, j) -> bool:
        """Merge across edge, then the same on the mirror side; False if either fails."""
        mi, mj = self.mesh.mirror_face.get(i), self.mesh.mirror_face.get(j)
        if not seg.merge(edge, i, j):
            return False
        if mi is None or mj is None:
            return False
        me = self.mesh.mirror_edge(edge, mi, mj)
        if me is None:
            return False
        if seg.piece[mi] == seg.piece[mj]:
            return True      # the mirror merge is this one (a piece across the centre line)
        return seg.merge(me, mi, mj)

    def greedy(self, seg: Segmentation, rng: np.random.Generator | None = None, pick: int = 1) -> Segmentation:
        """Merge while the score drops; each step takes the best merge, or (with `rng`) one of
        the `pick` best, so restarts explore other answers."""
        score, _ = seg.score()
        while True:
            found = []
            for options in self.candidates(seg):
                for edge, i, j in options:          # longest shared edge as hinge first
                    trial = seg.copy()
                    if self._mirrored(trial, edge, i, j):
                        s, _ = trial.score()
                        if s < score - 1e-6:
                            found.append((s, len(found), trial))
                        break
            if not found:
                return seg
            found.sort()
            k = int(rng.integers(min(pick, len(found)))) if rng is not None else 0
            score, _, seg = found[k]

    def run(self, seg: Segmentation, restarts: int = 0, seed: int = 1, log=None) -> Segmentation:
        """The plain greedy answer, then `restarts` randomised ones; the best score wins."""
        best = self.greedy(seg.copy())
        if log:
            log(f"  greedy    {best.score()[0]:7.1f}  {best.score()[1]}")
        rng = np.random.default_rng(seed)
        for n in range(restarts):
            trial = self.greedy(seg.copy(), rng, pick=3)
            if log:
                log(f"  restart {n + 1:2d} {trial.score()[0]:7.1f}  {trial.score()[1]}")
            if trial.score()[0] < best.score()[0]:
                best = trial
        return best


class Report:
    """CSV tables, a flat layout SVG and a 3D colour preview of a segmentation."""

    def __init__(self, mesh: Mesh, seg: Segmentation, names: dict[str, str]):
        self.mesh, self.seg, self.names = mesh, seg, names
        self.load = SeamLoad(seg.fold3d()).stress
        ranked = sorted(self.load, key=self.load.get, reverse=True)
        self.rank = {k: (n + 1) / len(ranked) for n, k in enumerate(ranked)}

    def pieces_csv(self, path: Path):
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["piece", "twin", "material", "faces", "area_mm2", "width_mm", "height_mm",
                        "pdf_pieces", "face_ids"])
            for label in sorted(self.seg.layout, key=self.names.get):
                lay, faces = self.seg.layout[label], self.seg.members(label)
                sides = lay.size()
                twin = self.seg.mirror_label(label)
                w.writerow([self.names[label], self.names.get(twin, ""),
                            self.mesh.faces[faces[0]].material, len(faces),
                            round(lay.area(), 1),
                            round(sides[0], 1), round(sides[1], 1),
                            " ".join(sorted({self.mesh.faces[f].piece for f in faces})),
                            " ".join(self.mesh.faces[f].id for f in faces)])

    def seams_csv(self, path: Path):
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["piece_a", "piece_b", "length_mm", "edges", "short", "stress", "stress_rank"])
            for r in sorted(self.seg.seams(), key=lambda r: r["length_mm"]):
                a, b = (list(r["pieces"]) * 2)[:2]
                key = frozenset(r["pieces"])
                w.writerow([self.names.get(a, a), self.names.get(b, b), round(r["length_mm"], 1),
                            len(r["edges"]), int(r["length_mm"] < TAPE_MM),
                            round(self.load.get(key, 0.0), 1),
                            round(self.rank[key], 3) if key in self.rank else ""])

    def svg(self, path: Path, colours: dict[str, str]):
        """Each piece laid flat in a row: cut outline solid, folds dashed. A dart (a slit
        the piece glues to itself) is cut on one side only; the slit between its two sides
        stays on the piece as a glue flap, with a red score line where the cut edge lands."""
        out, x, y, row_h = [], 10.0, 10.0, 0.0
        width = 400.0
        for label in sorted(self.seg.layout, key=self.names.get):
            lay = self.seg.layout[label]
            minx, miny, maxx, maxy = lay.outline().bounds
            if x + maxx - minx > width:
                x, y, row_h = 10.0, y + row_h + 15, 0.0
            dx, dy = x - minx, y - miny
            colour = colours.get(self.mesh.faces[self.seg.members(label)[0]].material, "#ccc")
            out.append(f'<g id="{self.names[label]}">')
            for i, p in lay.placed.items():
                pts = " ".join(f"{a + dx:.2f},{b + dy:.2f}" for a, b in p)
                out.append(f'<polygon points="{pts}" fill="{colour}" fill-opacity="0.35" stroke="none"/>')
            for i, p in lay.placed.items():
                for k, a, b in self.mesh.faces[i].edges():
                    edge = frozenset((a, b))
                    kind = self.seg.edge_kind(edge)
                    p0, p1 = p[k], p[(k + 1) % len(p)]
                    if kind == "dart" and i != min(f for f, _ in self.mesh.edge_faces[edge]):
                        # the dart's second side: kept as a glue flap filling the slit, scored
                        # where the cut side lands, so folding closes the dart onto the flap
                        other = min(f for f, _ in self.mesh.edge_faces[edge])
                        q = lay.placed[other]
                        q0, q1 = lay.corner(other, a), lay.corner(other, b)
                        flap = Polygon([p0, p1, q1, q0]).buffer(0)   # the slit itself: glued under the cut side
                        for g in getattr(flap, "geoms", [flap]):
                            if g.geom_type == "Polygon" and g.area > 1:
                                pts = " ".join(f"{u + dx:.2f},{v + dy:.2f}" for u, v in g.exterior.coords)
                                out.append(f'<polygon points="{pts}" fill="{colour}" fill-opacity="0.15" '
                                           f'stroke="#000" stroke-width="0.4"/>')
                        kind = "score"
                    style = {"fold": 'stroke="#555" stroke-dasharray="2,1.5"',
                             "score": 'stroke="#d00" stroke-dasharray="1,1"',
                             "dart": 'stroke="#000"'}.get(kind, 'stroke="#000"')
                    out.append(f'<line x1="{p0[0] + dx:.2f}" y1="{p0[1] + dy:.2f}" x2="{p1[0] + dx:.2f}" '
                               f'y2="{p1[1] + dy:.2f}" {style} stroke-width="0.4"/>')
            c = lay.outline().representative_point()
            out.append(f'<text x="{c.x + dx:.1f}" y="{c.y + dy:.1f}" font-size="5" '
                       f'font-family="sans-serif" text-anchor="middle">{self.names[label]}</text></g>')
            x += maxx - minx + 15
            row_h = max(row_h, maxy - miny)
        h = y + row_h + 10
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 20}mm" height="{h:.0f}mm" '
                        f'viewBox="0 0 {width + 20} {h:.0f}">\n' + "\n".join(out) + "\n</svg>\n")

    def nest_svg(self, path: Path, page: int = 90, depth: float = 7.5):
        """The pieces as a minimal lowpoly-export/1 file for papercraft-nester's `nest`: one
        group per piece (ids piece-<page>-NN, cardstock in data-material) with its cut
        outline, a 7.5 mm tab on one side of every seam (the piece whose name sorts first),
        and dart flaps. Enough to count sheets; not a cutting file."""
        out = []
        for n, label in enumerate(sorted(self.seg.layout, key=self.names.get), start=1):
            lay = self.seg.layout[label]
            shape = lay.outline()
            for i, p in lay.placed.items():
                for k, a, b in self.mesh.faces[i].edges():
                    edge = frozenset((a, b))
                    faces = self.mesh.edge_faces[edge]
                    if len(faces) < 2:
                        continue
                    other = next(f for f, _ in faces if f != i)
                    q = self.seg.piece[other]
                    mine = self.names.get(label, label) < self.names.get(q, q)
                    if self.seg.edge_kind(edge) == "seam" and mine:
                        shape = shape.union(self._flap(p[k], p[(k + 1) % len(p)], p.mean(axis=0), depth)
                                            .difference(lay.outline()))
                    elif self.seg.edge_kind(edge) == "dart" and i != min(f for f, _ in faces):
                        q0, q1 = lay.corner(other, a), lay.corner(other, b)
                        shape = shape.union(Polygon([p[k], p[(k + 1) % len(p)], q1, q0]).buffer(0))
            shape = shape.buffer(0.05).buffer(-0.05)
            if shape.geom_type != "Polygon":
                shape = max(shape.geoms, key=lambda g: g.area)
            xy = np.array(shape.exterior.coords)[:-1]
            d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in xy) + " Z"
            material = self.mesh.faces[self.seg.members(label)[0]].material
            out.append(f'<g id="piece-{page:02d}-{n:02d}" data-piece="piece-{page:02d}-{n:02d}" '
                       f'data-material="{material}" data-name="{self.names[label]}">'
                       f'<path data-operation="cut" d="{d}"/></g>')
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" data-contract="lowpoly-export/1">\n'
                        + "\n".join(out) + "\n</svg>\n")

    @staticmethod
    def _flap(p0: np.ndarray, p1: np.ndarray, inside: np.ndarray, depth: float = 7.5) -> Polygon:
        """A tab on edge p0-p1, on the side away from `inside`, ends sloped at 45 degrees."""
        d = p1 - p0
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
        if (inside - p0) @ n > 0:
            n = -n
        inset = min(depth, np.linalg.norm(d) / 2) * d / (np.linalg.norm(d) or 1.0)
        return Polygon([p0, p1, p1 - inset + depth * n, p0 + inset + depth * n])

    @staticmethod
    def preview(mesh: Mesh, segs: list[tuple[str, Segmentation]], path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        region = segs[0][1].region
        pts = mesh.points[sorted({v for i in region for v in mesh.faces[i].corners})]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        fig = plt.figure(figsize=(6 * len(segs), 6))
        cmap = plt.get_cmap("tab20")
        for n, (title, seg) in enumerate(segs):
            ax = fig.add_subplot(1, len(segs), n + 1, projection="3d")
            labels = sorted({seg.piece[i] for i in region})
            colour = {lab: cmap(k % 20) for k, lab in enumerate(labels)}
            polys = [mesh.points[mesh.faces[i].corners] for i in region]
            ax.add_collection3d(Poly3DCollection(polys, facecolors=[colour[seg.piece[i]] for i in region],
                                                 edgecolors="none", alpha=0.9))
            cuts, folds = [], []
            for edge, faces in mesh.edge_faces.items():
                if any(f in region for f, _ in faces):
                    (cuts if seg.edge_kind(edge) != "fold" else folds).append(mesh.points[list(edge)])
            ax.add_collection3d(Line3DCollection(folds, colors="#777", linewidths=0.4, linestyles="dashed"))
            ax.add_collection3d(Line3DCollection(cuts, colors="black", linewidths=1.4))
            ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
            ax.view_init(elev=15, azim=-60)
            ax.set_axis_off()
            _, parts = seg.score()
            ax.set_title(f"{title}\n{parts['pieces']} pieces, {parts['seams']} seams, "
                         f"{parts['short_seams']} under {TAPE_MM:.0f} mm", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def _names(seg: Segmentation, prefix: str) -> dict[str, str]:
    """Readable piece names: <prefix>-01, -02 ..., a twin right after its twin (L/R)."""
    names, n = {}, 0
    order = sorted(seg.layout, key=lambda lab: tuple(-seg.mesh.centre(seg.members(lab)[0])[[2, 1]]))
    for label in order:
        if label in names:
            continue
        n += 1
        twin = seg.mirror_label(label)
        if twin is None or twin == label:
            names[label] = f"{prefix}-{n:02d}"
        else:
            right = seg.mesh.centre(seg.members(label)[0])[0] > seg.mesh.centre(seg.members(twin)[0])[0]
            names[label], names[twin] = (f"{prefix}-{n:02d}{'R' if right else 'L'}",
                                         f"{prefix}-{n:02d}{'L' if right else 'R'}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold3d", type=Path, required=True, help="data/<project>/fold3d.json")
    ap.add_argument("--pieces", default="", help="PDF pieces to re-cut (their twins are added), comma separated")
    ap.add_argument("--material", default="", help="or: every piece of this cardstock")
    ap.add_argument("--name", default="segment", help="output name and new piece prefix")
    ap.add_argument("--restarts", type=int, default=8, help="randomised greedy runs after the plain one")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-shapes", action="store_true",
                    help="keep the PDF's face shapes (default: make the mesh exactly symmetric first)")
    ap.add_argument("--out", type=Path, help="output folder (default: data/<project>/segment/)")
    args = ap.parse_args()
    data = json.loads(args.fold3d.read_text(encoding="utf-8"))
    mesh = Mesh(data)
    if not args.keep_shapes:
        print(f"symmetric mesh: corners moved up to {mesh.symmetrize():.1f} mm")
    chosen = {p for p in args.pieces.split(",") if p}
    chosen |= {p for p, v in data["pieces"].items() if args.material and v["material"] == args.material}
    chosen |= {data["pieces"][p].get("twin") for p in chosen} - {None, ""}
    region = {i for i, f in enumerate(mesh.faces) if f.piece in chosen}
    if not region:
        ap.error("no faces: give --pieces or --material")
    out = args.out or args.fold3d.parent / "segment"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(region)} faces of {len(chosen)} PDF pieces")

    pdf = Segmentation.from_pieces(mesh, region)
    print(f"PDF pieces  {pdf.score()[1]}")
    new = Segmenter(mesh, region).run(Segmentation(mesh, region), args.restarts, args.seed, log=print)
    print(f"re-cut      {new.score()[1]}")
    colours = data.get("colours", {})
    for tag, seg, prefix in (("pdf", pdf, None), ("new", new, args.name)):
        names = ({lab: mesh.faces[seg.members(lab)[0]].piece for lab in seg.layout} if prefix is None
                 else _names(seg, prefix))
        rep = Report(mesh, seg, names)
        rep.pieces_csv(out / f"{args.name}-{tag}-pieces.csv")
        rep.seams_csv(out / f"{args.name}-{tag}-seams.csv")
        rep.svg(out / f"{args.name}-{tag}.svg", colours)
    Report.preview(mesh, [("PDF pieces", pdf), ("re-cut from the mesh", new)], out / f"{args.name}.png")
    print(f"wrote {out}/{args.name}-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
