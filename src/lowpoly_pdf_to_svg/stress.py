"""Which seams carry the most load once the model is worn, from the folded 3D model.

A mask rests on the head at its largest opening (the neck). "Up" runs from there to the
model's centre of mass. Each piece weighs its paper area. Load reaches the neck through
the strongest seams: a shortest-path tree from the neck pieces, where a seam costs one over
its glued length. A seam holds everything beyond it in that tree, and its stress is

    mass beyond x sideways lever arm (centre of that mass from the seam, across "up")
    / glued length squared

so a nose or a horn hanging off a short seam scores high. This is a tree estimate: a real
model shares load over its loops, so read it as a ranking, not a force.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLUSTER_MM = 1.0   # 3D corners this close are one corner of the folded model


class SeamLoad:
    def __init__(self, fold3d: Path):
        data = json.loads(fold3d.read_text(encoding="utf-8"))
        self.points = np.array(data["vertices"], float)
        self.faces, self.glues = data["faces"], data["glues"]
        self.corner = self._clusters()
        self.area, self.moment = self._weights()
        self.stress: dict[frozenset, float] = self._stress()

    def _clusters(self) -> list[int]:
        parent = list(range(len(self.points)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        keys = np.floor(self.points / CLUSTER_MM).astype(int)
        cells = defaultdict(list)
        for i, key in enumerate(map(tuple, keys)):
            cells[key].append(i)
        for i, (x, y, z) in enumerate(keys):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((x + dx, y + dy, z + dz), ()):
                            if j > i and np.linalg.norm(self.points[i] - self.points[j]) <= CLUSTER_MM:
                                parent[find(i)] = find(j)
        return [find(i) for i in range(len(self.points))]

    def _weights(self):
        area, moment = defaultdict(float), defaultdict(lambda: np.zeros(3))
        for face in self.faces:
            p = self.points[face["v"]]
            for k in range(1, len(p) - 1):
                a = np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) / 2
                area[face["piece"]] += a
                moment[face["piece"]] += a * (p[0] + p[k] + p[k + 1]) / 3
        return area, moment

    def _neck(self) -> tuple[set[str], np.ndarray]:
        """Pieces on the longest rim loop, and that loop's centre."""
        used, owner = Counter(), {}
        for face in self.faces:
            v = face["v"]
            for k in range(len(v)):
                edge = frozenset((v[k], v[(k + 1) % len(v)]))
                used[edge] += 1
                owner[edge] = face["piece"]
        glued = {frozenset(g["v"][:2]) for g in self.glues} | {frozenset(g["v"][2:]) for g in self.glues}
        rim = [e for e, n in used.items() if n == 1 and e not in glued and len(e) == 2]
        root = {}

        def find(x):
            while root.setdefault(x, x) != x:
                x = root[x]
            return x
        for edge in rim:
            a, b = (self.corner[i] for i in edge)
            root[find(a)] = find(b)
        loops = defaultdict(list)
        for edge in rim:
            loops[find(self.corner[next(iter(edge))])].append(edge)
        if not loops:
            return set(), np.zeros(3)
        length = lambda es: sum(np.linalg.norm(np.subtract(*self.points[list(e)])) for e in es)
        neck = max(loops.values(), key=length)
        return {owner[e] for e in neck}, np.mean([self.points[i] for e in neck for i in e], axis=0)

    def _stress(self) -> dict[frozenset, float]:
        support, neck = self._neck()
        total = sum(self.area.values())
        if not support or not total:
            return {}
        up = sum(self.moment.values()) / total - neck
        up /= np.linalg.norm(up) or 1.0
        seams = defaultdict(lambda: [0.0, np.zeros(3)])
        for g in self.glues:
            if "closes" not in g["status"] or g["piece"] == g["tab_piece"]:
                continue
            a, b = self.points[g["v"][0]], self.points[g["v"][1]]
            seam = seams[frozenset((g["piece"], g["tab_piece"]))]
            length = float(np.linalg.norm(a - b))
            seam[0] += length
            seam[1] += length * (a + b) / 2
        links = defaultdict(list)
        for key in seams:
            p, q = tuple(key)
            links[p].append((q, key))
            links[q].append((p, key))
        cost, via = {p: 0.0 for p in support}, {}
        queue = [(0.0, p) for p in sorted(support)]
        while queue:
            c, p = heapq.heappop(queue)
            if c > cost.get(p, np.inf):
                continue
            for q, key in links[p]:
                if c + 1 / seams[key][0] < cost.get(q, np.inf):
                    cost[q], via[q] = c + 1 / seams[key][0], key
                    heapq.heappush(queue, (cost[q], q))
        children = defaultdict(list)
        for q, key in via.items():
            children[next(p for p in key if p != q)].append(q)
        held = {}
        order = sorted(cost, key=cost.get, reverse=True)   # leaves first
        for p in order:
            mass, moment = self.area[p], self.moment[p].copy()
            for q in children[p]:
                mass, moment = mass + held[q][0], moment + held[q][1]
            held[p] = (mass, moment)
        stress = {}
        for q, key in via.items():
            mass, moment = held[q]
            length, middle = seams[key][0], seams[key][1] / seams[key][0]
            arm = moment / mass - middle
            stress[key] = float(mass * np.linalg.norm(arm - (arm @ up) * up) / length ** 2)
        return stress

    def top(self, share: float) -> set[frozenset]:
        """The seams in the top `share` by stress."""
        ranked = sorted(self.stress, key=self.stress.get, reverse=True)
        return set(ranked[:round(share * len(ranked))])
