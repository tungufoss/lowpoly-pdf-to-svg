"""Choose the best set of pieces exactly, by set partitioning (lowpoly #68).

segment.py's greedy merge stops at the first answer it cannot improve one merge at a time.
Here a pool of candidate pieces is gathered (every piece the greedy runs and their
randomised restarts build, every single face, and each of those grown by one neighbouring
face), all already known to lie flat without overlap and to fit the sheet. A mixed-integer
program then picks pieces so that every face of the region is in exactly one, and a piece
is picked together with its mirror twin.

The score is segment.py's, made linear: each piece costs its own (a piece, a tiny piece),
and each mesh edge costs what it would as a seam (the 15 mm tape rule, length) unless the
picked piece holding both of its faces folds it, so a piece is credited with the seam cost
of every edge it folds. Edges on the region's border cost the same in every answer and are
left out.

Solvers: HiGHS through SciPy (default, no licence) or Gurobi (`--solver gurobi`; the free
pip licence takes 2000 variables, enough for the nose; bigger regions need a full licence).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from . import segment
from .palette import yaml_colours
from .segment import (HINGE_MM, STRAIGHT_DEG, TAPE_MM, TINY_MM2, Layout, Mesh, Report, Segmentation,
                      Segmenter, _names)


class CandidatePool:
    """Pieces that lie flat and fit the sheet, keyed by their set of faces."""

    def __init__(self, mesh: Mesh, region: set[int]):
        self.mesh, self.region = mesh, region
        self.pieces: dict[frozenset, Layout] = {frozenset([i]): Layout.single(mesh, i) for i in region}

    def from_greedy(self, restarts: int, seed: int) -> Segmentation:
        """Every piece built along the greedy run and the restarts; returns the greedy's best."""
        seg = Segmentation(self.mesh, self.region)
        seg.pool = self.pieces
        return Segmenter(self.mesh, self.region).run(seg, restarts, seed)

    def grow(self, limit: int = 20000):
        """Each piece plus one neighbouring face of its cardstock, where it still lies flat."""
        for faces, lay in list(self.pieces.items()):
            if len(self.pieces) >= limit:
                return
            for i in faces:
                for k, a, b in self.mesh.faces[i].edges():
                    edge = frozenset((a, b))
                    for g, _ in self.mesh.edge_faces[edge]:
                        if g in faces or g not in self.region or \
                                self.mesh.faces[g].material != self.mesh.faces[i].material:
                            continue
                        key = faces | {g}
                        if key in self.pieces:
                            continue
                        joined = lay.joined(Layout.single(self.mesh, g), edge, i, g)
                        if joined is not None and joined.fits():
                            self.pieces[key] = joined

    def mirror(self, faces: frozenset) -> frozenset | None:
        m = [self.mesh.mirror_face.get(f) for f in faces]
        return None if None in m else frozenset(m)

    def paired(self) -> list[frozenset]:
        """The pieces whose mirror twin is in the pool too (or that are their own twin)."""
        return [f for f in self.pieces if self.mirror(f) in self.pieces]


class PartitionModel:
    def __init__(self, mesh: Mesh, region: set[int], pool: CandidatePool):
        self.mesh, self.region, self.pool = mesh, region, pool
        self.keys = pool.paired()
        self.edge_cost = {e: self._seam_cost(e) for e, fs in mesh.edge_faces.items()
                          if len(fs) == 2 and all(f in region for f, _ in fs)}

    def _seam_cost(self, edge: frozenset) -> float:
        """segment.py's seam terms for one edge: a seam, its length, and short if no straight
        neighbour could extend it to a tape strip."""
        length = self.mesh.length(edge)
        short = length < TAPE_MM and not any(self._straight(edge, e) for v in edge
                                             for e in self._at(v) if e != edge)
        return 5 + length / 10 + 100 * short

    def _at(self, v: int):
        if not hasattr(self, "_vertex_edges"):
            self._vertex_edges = {}
            for e in self.mesh.edge_faces:
                for w in e:
                    self._vertex_edges.setdefault(w, []).append(e)
        return self._vertex_edges.get(v, [])

    def _straight(self, e1: frozenset, e2: frozenset) -> bool:
        shared = e1 & e2
        if len(shared) != 1:
            return False
        (s,) = shared
        (a,), (b,) = e1 - shared, e2 - shared
        u, v = self.mesh.points[s] - self.mesh.points[a], self.mesh.points[b] - self.mesh.points[s]
        return u @ v >= math.cos(math.radians(STRAIGHT_DEG)) * np.linalg.norm(u) * np.linalg.norm(v)

    def cost(self, faces: frozenset) -> float:
        lay = self.pool.pieces[faces]
        c = 20 + 60 * (lay.area() < TINY_MM2) + segment.BOX_WEIGHT * lay.box_area()
        for i in faces:
            for _, a, b in self.mesh.faces[i].edges():
                edge = frozenset((a, b))
                (f, _), (g, _) = (self.mesh.edge_faces[edge] + [(None, None)])[:2]
                other = g if f == i else f
                if edge in self.edge_cost and other in faces and i < other:
                    gap = lay.gap(edge, i, other)
                    if gap <= HINGE_MM:
                        c -= self.edge_cost[edge]        # folded: no seam
                    elif gap < segment.DART_MM:
                        c += 100                         # a dart too thin to glue
        return c

    def solve(self, solver: str = "highs", time_limit: float = 120.0) -> list[Layout]:
        keys, faces = self.keys, sorted(self.region)
        col = {k: n for n, k in enumerate(keys)}
        c = np.array([self.cost(k) for k in keys])
        cover = [[col[k] for k in keys if f in k] for f in faces]
        twins = {(col[k], col[self.pool.mirror(k)]) for k in keys if self.pool.mirror(k) != k}
        twins = {tuple(sorted(t)) for t in twins}
        if solver == "gurobi":
            chosen = self._gurobi(c, cover, twins, time_limit)
        else:
            chosen = self._highs(c, cover, twins, time_limit)
        return [self.pool.pieces[keys[n]] for n in chosen]

    @staticmethod
    def _highs(c, cover, twins, time_limit):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
        rows = len(cover) + len(twins)
        a = lil_matrix((rows, len(c)))
        lo, hi = np.zeros(rows), np.zeros(rows)
        for r, cols in enumerate(cover):
            a[r, cols] = 1
            lo[r] = hi[r] = 1
        for r, (p, q) in enumerate(twins, start=len(cover)):
            a[r, p], a[r, q] = 1, -1
        res = milp(c, constraints=LinearConstraint(a.tocsr(), lo, hi), integrality=np.ones(len(c)),
                   bounds=Bounds(0, 1), options={"time_limit": time_limit})
        if res.x is None:
            raise RuntimeError(f"no partition found: {res.message}")
        return [n for n, v in enumerate(res.x) if v > 0.5]

    @staticmethod
    def _gurobi(c, cover, twins, time_limit):
        import gurobipy as gp
        m = gp.Model("partition")
        m.Params.OutputFlag, m.Params.TimeLimit = 0, time_limit
        x = m.addVars(len(c), vtype=gp.GRB.BINARY, obj=list(c))
        for cols in cover:
            m.addConstr(gp.quicksum(x[n] for n in cols) == 1)
        for p, q in twins:
            m.addConstr(x[p] == x[q])
        m.optimize()
        return [n for n in range(len(c)) if x[n].X > 0.5]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold3d", type=Path, required=True)
    ap.add_argument("--pieces", default="")
    ap.add_argument("--material", default="")
    ap.add_argument("--name", default="partition")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--grow", type=int, default=1, help="rounds of growing pool pieces by one face")
    ap.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dart-mm", type=float, default=0.0,
                    help="a dart whose slit is narrower than this counts as a short seam (0: thin darts "
                         "are fine, e.g. taped from inside; 7.5: avoid them)")
    ap.add_argument("--max-piece", default="", help="largest piece WxH in mm (default: one sheet less a 15 mm border)")
    ap.add_argument("--sheet-cost", type=float, default=20.0,
                    help="cost of one sheet's worth of bounding boxes, against 20 per piece")
    ap.add_argument("--flatten-deg", type=float, default=0.0,
                    help="geometry edit: make vertices with an angle defect under this flat (0: off)")
    ap.add_argument("--flatten-mm", type=float, default=3.0, help="... moving a vertex at most this far")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    segment.DART_MM = args.dart_mm
    segment.BOX_WEIGHT = args.sheet_cost / segment.SHEET_AREA
    if args.max_piece:
        segment.MAX_PIECE_MM = tuple(float(v) for v in args.max_piece.lower().split("x"))
    data = json.loads(args.fold3d.read_text(encoding="utf-8"))
    yaml_colours(data, args.fold3d)   # faces recoloured in the viewer split their piece
    mesh = Mesh(data)
    mesh.symmetrize()
    chosen = {p for p in args.pieces.split(",") if p}
    chosen |= {p for p, v in data["pieces"].items() if args.material and v["material"] == args.material}
    chosen |= {data["pieces"][p].get("twin") for p in chosen} - {None, ""}
    region = {i for i, f in enumerate(mesh.faces) if f.piece in chosen}
    if not region:
        ap.error("no faces: give --pieces or --material")
    out = args.out or args.fold3d.parent / "segment"
    out.mkdir(parents=True, exist_ok=True)
    if args.flatten_deg > 0:
        edits = mesh.flatten_vertices(region, args.flatten_deg, args.flatten_mm)
        with (out / f"{args.name}-flattened.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, ["vertex", "defect_deg", "moved_mm"])
            w.writeheader()
            w.writerows(edits)
        print(f"flattened {len(edits)} nearly flat vertices, moved up to "
              f"{max([e['moved_mm'] for e in edits], default=0):.1f} mm")

    t = time.time()
    pool = CandidatePool(mesh, region)
    greedy = pool.from_greedy(args.restarts, args.seed)
    for _ in range(args.grow):
        pool.grow()
    model = PartitionModel(mesh, region, pool)
    print(f"{len(region)} faces, {len(pool.pieces)} candidate pieces ({len(model.keys)} with twin), "
          f"{time.time() - t:.0f} s")
    t = time.time()
    best = Segmentation.from_layouts(mesh, region, model.solve(args.solver))
    print(f"solved with {args.solver} in {time.time() - t:.1f} s")
    print(f"greedy     {greedy.score()[1]}")
    print(f"partition  {best.score()[1]}")
    if best.score()[0] > greedy.score()[0]:
        print("the greedy answer scores better (the linear score only approximates seam runs); keeping it")
        best = greedy
    rep = Report(mesh, best, _names(best, args.name))
    rep.pieces_csv(out / f"{args.name}-pieces.csv")
    rep.seams_csv(out / f"{args.name}-seams.csv")
    rep.svg(out / f"{args.name}.svg", data.get("colours", {}))
    rep.nest_svg(out / f"{args.name}-nest.svg")
    pdf = Segmentation.from_pieces(mesh, region)
    Report(mesh, pdf, {lab: mesh.faces[pdf.members(lab)[0]].piece for lab in pdf.layout}).nest_svg(
        out / f"{args.name}-pdf-nest.svg", page=91)
    Report.preview(mesh, [("greedy", greedy), ("set partition", best)], out / f"{args.name}.png")
    print(f"wrote {out}/{args.name}-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
