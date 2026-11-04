"""Build order: which piece to glue first, from the folded 3D model (fold3d.json).

A mask is glued from the inside, so every seam needs a hand behind it. The way in is the
neck, the largest opening (`SeamLoad._neck`). The order is a sweep from the pieces farthest
from the neck towards it: a piece goes on once everything farther away is on, so the part
of the model between it and the neck is still open and its seams can be reached. The neck
rim goes on last. Far-away parts that do not touch yet (a horn tip, a hair spike) start
their own sub-assembly and are joined when the sweep reaches them; the step says so.

Distance is measured along the model: a shortest path from the neck pieces over the glued
seams, each seam costing the 3D distance between the two pieces' centres. Mirror twins go
on in one step and share its number, so a step is one number on the Cricut labels whatever
the side. Among pieces nearly as far as the farthest (`SLACK_MM`), one that glues onto the
pieces already on wins, so the build stays in one place instead of hopping round the model.

    python -m src.lowpoly_pdf_to_svg.assembly --fold3d data/<p>/fold3d.json
writes data/<p>/assembly.json (steps for the viewer's Assembly player) and assembly.csv.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stress import SeamLoad

SLACK_MM = 25.0   # pieces this close to the farthest one count as equally far


@dataclass
class Step:
    number: int
    pieces: list[str]
    distance_mm: float
    onto: list[str] = field(default_factory=list)    # pieces already on that this step glues to
    starts_part: bool = False                          # nothing to glue onto yet: a new sub-assembly
    joins: int = 0                                     # sub-assemblies this step joins into one

    def row(self, materials: dict[str, str]) -> dict:
        return {"step": self.number, "pieces": self.pieces,
                "material": materials.get(self.pieces[0], ""),
                "distance_mm": round(self.distance_mm, 1), "onto": self.onto,
                "starts_part": self.starts_part, "joins": self.joins}


class AssemblyOrder:
    def __init__(self, fold3d: Path | dict, slack_mm: float = SLACK_MM):
        data = fold3d if isinstance(fold3d, dict) else json.loads(fold3d.read_text(encoding="utf-8"))
        self.load = SeamLoad(data)
        self.materials = {p: str(info.get("material", "")) for p, info in data.get("pieces", {}).items()}
        twins = {p: info.get("twin") for p, info in data.get("pieces", {}).items() if info.get("twin")}
        pieces = sorted(self.load.area)
        self.centre = {p: self.load.moment[p] / self.load.area[p] for p in pieces if self.load.area[p]}
        self.links = self._links()
        self.distance = self._distances()
        self.groups = self._groups(pieces, twins)
        self.slack = slack_mm
        self.steps = self._sweep()

    def _links(self) -> dict[str, set[str]]:
        links = defaultdict(set)
        for g in self.load.glues:
            p, q = g["piece"], g["tab_piece"]
            if p != q and "closes" in g["status"]:
                links[p].add(q)
                links[q].add(p)
        return links

    def _distances(self) -> dict[str, float]:
        neck, _ = self.load._neck()
        dist = {p: 0.0 for p in neck if p in self.centre}
        queue = [(0.0, p) for p in sorted(dist)]
        while queue:
            d, p = heapq.heappop(queue)
            if d > dist[p]:
                continue
            for q in self.links[p]:
                if q not in self.centre:
                    continue
                nd = d + float(np.linalg.norm(self.centre[p] - self.centre[q]))
                if nd < dist.get(q, np.inf):
                    dist[q] = nd
                    heapq.heappush(queue, (nd, q))
        far = max(dist.values(), default=0.0)
        return {p: dist.get(p, far) for p in self.centre}   # unreachable pieces: as far as the farthest

    @staticmethod
    def _groups(pieces: list[str], twins: dict[str, str]) -> list[list[str]]:
        seen, groups = set(), []
        for p in pieces:
            if p in seen:
                continue
            group = [p] + ([twins[p]] if twins.get(p) in pieces and twins[p] not in seen else [])
            seen.update(group)
            groups.append(group)
        return groups

    def _sweep(self) -> list[Step]:
        placed: set[str] = set()
        part: dict[str, int] = {}          # piece -> sub-assembly id
        left = list(self.groups)
        steps = []
        far = lambda g: max(self.distance.get(p, 0.0) for p in g)
        touches = lambda g: any(q in placed for p in g for q in self.links[p])
        while left:
            top = max(far(g) for g in left)
            near = [g for g in left if far(g) >= top - self.slack]
            # Prefer a group gluing onto the last step, then onto anything placed, then the farthest.
            last = set(steps[-1].pieces) if steps else set()
            group = max(near, key=lambda g: (any(q in last for p in g for q in self.links[p]), touches(g), far(g), [-ord(c) for c in g[0]]))
            left.remove(group)
            onto = sorted({q for p in group for q in self.links[p] if q in placed})
            parts = {part[q] for q in onto}
            new = min(parts) if parts else len(steps)
            for p, k in list(part.items()):
                if k in parts:
                    part[p] = new
            for p in group:
                part[p] = new
            placed.update(group)
            steps.append(Step(len(steps) + 1, group, far(group), onto, not onto, max(0, len(parts) - 1)))
        return steps

    def order(self) -> list[str]:
        """Pieces in build order (twins side by side)."""
        return [p for step in self.steps for p in step.pieces]

    def write(self, folder: Path) -> None:
        rows = [s.row(self.materials) for s in self.steps]
        (folder / "assembly.json").write_text(json.dumps({"rule": "far from the neck first; twins share a step",
                                                           "steps": rows}, indent=1) + "\n", encoding="utf-8")
        with (folder / "assembly.csv").open("w", newline="", encoding="utf-8") as handle:
            out = csv.writer(handle)
            out.writerow(["step", "pieces", "material", "distance_mm", "onto", "starts_part", "joins"])
            for r in rows:
                out.writerow([r["step"], " ".join(r["pieces"]), r["material"], r["distance_mm"],
                              " ".join(r["onto"]), int(r["starts_part"]), r["joins"]])

    @staticmethod
    def read(path: Path) -> list[str]:
        """Build order from a written assembly.json (absent: empty)."""
        if not path.is_file():
            return []
        return [p for step in json.loads(path.read_text(encoding="utf-8"))["steps"] for p in step["pieces"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build order from the folded model: far from the neck first.")
    parser.add_argument("--fold3d", type=Path, required=True, help="fold3d.json of the model")
    parser.add_argument("--slack-mm", type=float, default=SLACK_MM, help="how much nearer a piece may be and still count as farthest")
    args = parser.parse_args()
    order = AssemblyOrder(args.fold3d, args.slack_mm)
    order.write(args.fold3d.parent)
    starts = sum(s.starts_part for s in order.steps)
    print(f"{args.fold3d.parent / 'assembly.json'}: {len(order.steps)} steps, {starts} start a sub-assembly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
