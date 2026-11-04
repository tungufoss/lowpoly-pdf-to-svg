"""Search pieces with an evolutionary algorithm, scored by seams and by sheets (lowpoly #68).

The set-partitioning solve (partition.py) is exact only over its pool of candidate pieces,
and it can count sheets only through bounding boxes. Here a random-key genetic algorithm
(after Takahashi et al. 2011, who stitch patches with a GA) searches the splits directly:

* genome: one number in [0, 1] per mirror pair of mesh edges inside the region;
* decoding: the edges whose key is above 0.5, from the highest key down, are folded
  (their two pieces merged, both twins at once) when the merged piece still lies flat
  without overlap and fits the sheet; the rest stay seams;
* fitness: segment.py's score plus a sheet term from a fast packing proxy (each piece's
  smallest rectangle, grown by a tab on two sides, packed shelf by shelf, tallest first,
  onto Letter sheets). The count is fractional (full sheets plus how full the last one
  is), so the search can feel its way down to one sheet fewer.

The population starts from random keys and from the greedy answer. Each generation keeps
the best two, and the rest are children of tournament winners (uniform crossover, then
each key redrawn with a small chance). The best few distinct splits are written for the
real nester (`<name>-<k>-nest.svg`) to confirm the sheet count.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import segment
from .segment import Mesh, Report, Segmentation, Segmenter, _names

LETTER_MM = (215.9, 279.4)
TAB_MM = 7.5

_STATE: dict = {}


class ShelfPacker:
    """Rectangles onto sheets, shelf by shelf, tallest first, each turned whichever way fits."""

    def __init__(self, sheet=LETTER_MM, gap: float = 1.0):
        self.w, self.h = sheet
        self.gap = gap

    def sheets(self, rects: list[tuple[float, float]]) -> float:
        items = sorted(((max(r), min(r)) for r in rects), reverse=True)
        sheets = []                      # each: list of shelves [y, height, x_used]
        for long, short in items:
            placed = False
            for shelves in sheets:
                for shelf in shelves:     # an existing shelf with room, piece lying flat
                    for w, h in ((long, short), (short, long)):
                        if h <= shelf[1] and shelf[2] + w <= self.w:
                            shelf[2] += w + self.gap
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
                top = shelves[-1][0] + shelves[-1][1] + self.gap if shelves else 0.0
                for w, h in ((long, short), (short, long)):
                    if w <= self.w and top + h <= self.h:
                        shelves.append([top, h, w + self.gap])
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                w, h = (long, short) if long <= self.w else (short, long)
                sheets.append([[0.0, h, w + self.gap]])
        if not sheets:
            return 0.0
        last = sheets[-1][-1]
        return len(sheets) - 1 + (last[0] + last[1]) / self.h


class Genome:
    """Mirror pairs of region edges, and how keys decode into a segmentation."""

    def __init__(self, mesh: Mesh, region: set[int]):
        self.mesh, self.region = mesh, region
        seen, self.classes = set(), []
        for edge, faces in mesh.edge_faces.items():
            if edge in seen or len(faces) != 2:
                continue
            (i, _), (j, _) = faces
            if i not in region or j not in region or mesh.faces[i].material != mesh.faces[j].material:
                continue
            mi, mj = mesh.mirror_face.get(i), mesh.mirror_face.get(j)
            twin = mesh.mirror_edge(edge, mi, mj) if mi is not None and mj is not None else None
            seen |= {edge, twin} - {None}
            self.classes.append((edge, i, j))
        self.segmenter = Segmenter(mesh, region)

    def decode(self, keys: np.ndarray) -> Segmentation:
        seg = Segmentation(self.mesh, self.region)
        for n in np.argsort(-keys):
            if keys[n] <= 0.5:
                break
            edge, i, j = self.classes[n]
            if seg.piece[i] == seg.piece[j]:
                continue
            trial = seg.copy()
            if self.segmenter._mirrored(trial, edge, i, j):
                seg = trial
        return seg

    def encode(self, seg: Segmentation) -> np.ndarray:
        """Keys that rebuild `seg`: its folded edges high (longest first), its seams low."""
        keys = np.full(len(self.classes), 0.2)
        for n, (edge, i, j) in enumerate(self.classes):
            if seg.piece[i] == seg.piece[j]:
                keys[n] = 0.6 + 0.4 * min(self.mesh.length(edge) / 100, 1)
        return keys


def fitness(seg: Segmentation, sheet_cost: float) -> tuple[float, float]:
    rects = [tuple(s + TAB_MM for s in seg.layout[lab].size()) for lab in seg.layout]
    sheets = ShelfPacker().sheets(rects)
    return seg.score()[0] + sheet_cost * sheets, sheets


def _init(fold3d: str, chosen: list[str], sheet_cost: float):
    data = json.loads(Path(fold3d).read_text(encoding="utf-8"))
    mesh = Mesh(data)
    mesh.symmetrize()
    region = {i for i, f in enumerate(mesh.faces) if f.piece in set(chosen)}
    _STATE.update(genome=Genome(mesh, region), sheet_cost=sheet_cost)


def _evaluate(keys: np.ndarray) -> tuple[float, float]:
    return fitness(_STATE["genome"].decode(keys), _STATE["sheet_cost"])


class Evolution:
    def __init__(self, genome: Genome, pool: Pool, size: int = 40, mutation: float = 0.05, seed: int = 1):
        self.genome, self.pool, self.size, self.mutation = genome, pool, size, mutation
        self.rng = np.random.default_rng(seed)

    def run(self, seeds: list[np.ndarray], generations: int, log=print):
        n = len(self.genome.classes)
        population = [s.copy() for s in seeds]
        population += [self.rng.random(n) for _ in range(self.size - len(population))]
        scores = self.pool.map(_evaluate, population)
        for g in range(generations):
            order = np.argsort([s[0] for s in scores])
            best = scores[order[0]]
            log(f"  generation {g:3d}  best {best[0]:7.1f}  sheets {best[1]:.2f}")
            children = [population[k] for k in order[:2]]
            while len(children) < self.size:
                a, b = (population[self._tournament(scores)] for _ in range(2))
                child = np.where(self.rng.random(n) < 0.5, a, b)
                redraw = self.rng.random(n) < self.mutation
                child[redraw] = self.rng.random(redraw.sum())
                children.append(child)
            population = children
            scores = [scores[k] for k in order[:2]] + self.pool.map(_evaluate, children[2:])
        order = np.argsort([s[0] for s in scores])
        return [(population[k], scores[k]) for k in order]

    def _tournament(self, scores, k: int = 3) -> int:
        picks = self.rng.integers(len(scores), size=k)
        return int(min(picks, key=lambda p: scores[p][0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold3d", type=Path, required=True)
    ap.add_argument("--pieces", default="")
    ap.add_argument("--material", default="")
    ap.add_argument("--name", default="evolve")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--sheet-cost", type=float, default=200.0, help="score of one Letter sheet (a piece is 20)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep", type=int, default=3, help="best distinct splits written for the nester")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    data = json.loads(args.fold3d.read_text(encoding="utf-8"))
    chosen = {p for p in args.pieces.split(",") if p}
    chosen |= {p for p, v in data["pieces"].items() if args.material and v["material"] == args.material}
    chosen |= {data["pieces"][p].get("twin") for p in chosen} - {None, ""}
    out = args.out or args.fold3d.parent / "segment"
    out.mkdir(parents=True, exist_ok=True)

    _init(str(args.fold3d), sorted(chosen), args.sheet_cost)
    genome = _STATE["genome"]
    mesh, region = genome.mesh, genome.region
    t = time.time()
    greedy = Segmenter(mesh, region).greedy(Segmentation(mesh, region))
    pdf = Segmentation.from_pieces(mesh, region)
    print(f"{len(region)} faces, {len(genome.classes)} edge pairs; greedy {fitness(greedy, args.sheet_cost)}, "
          f"PDF {fitness(pdf, args.sheet_cost)}")
    with Pool(args.workers, _init, (str(args.fold3d), sorted(chosen), args.sheet_cost)) as pool:
        ranked = Evolution(genome, pool, args.population, seed=args.seed).run(
            [genome.encode(greedy), genome.encode(pdf)], args.generations)
    print(f"{time.time() - t:.0f} s")
    seen, k = set(), 0
    for keys, (score, sheets) in ranked:
        seg = genome.decode(keys)
        key = frozenset(frozenset(lay.placed) for lay in seg.layout.values())
        if key in seen:
            continue
        seen.add(key)
        k += 1
        print(f"#{k}: fitness {score:.1f}, proxy sheets {sheets:.2f}, {seg.score()[1]}")
        rep = Report(mesh, seg, _names(seg, f"{args.name}{k}"))
        rep.pieces_csv(out / f"{args.name}-{k}-pieces.csv")
        rep.svg(out / f"{args.name}-{k}.svg", data.get("colours", {}))
        rep.nest_svg(out / f"{args.name}-{k}-nest.svg")
        if k >= args.keep:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
