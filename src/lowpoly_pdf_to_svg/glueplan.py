"""Modular glue plan: cardstock colour -> glue units -> joins between colours.

A glue unit is a set of same-colour pieces tied together by numbered glues (a tab onto
a face); a piece with no same-colour glue is a unit of its own. Units are glued first,
each on its own, and then joined to the units of other colours. fold3d.py folds the
units in that order; this module only names them and writes the plan.

    python -m src.lowpoly_pdf_to_svg.glueplan --config data/example/example.yaml

reads <project>.svg (no solve, seconds) and, when present, the statuses in the last
fold3d-glue.csv, and writes fold3d-units.csv and fold3d-joins.csv.
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GlueUnit:
    id: str                       # <colour>-NN, numbered by size within its colour
    material: str
    pieces: list[str]
    glues: list[int] = field(default_factory=list)   # indices into GluePlan.glues


@dataclass
class UnitJoin:
    a: str                        # unit ids, a < b
    b: str
    glues: list[int] = field(default_factory=list)


class GluePlan:
    """Partition pieces by colour into glue units and list the glues between units.

    `glues` are (number, piece, tab_piece) triples; any object with those attributes
    (fold3d.Glue) or tuples work."""

    def __init__(self, materials: dict[str, str], glues):
        self.materials = materials
        self.glues = [self._triple(g) for g in glues]
        self.units = self._units()
        self.unit_of = {piece: unit.id for unit in self.units for piece in unit.pieces}
        self.joins = self._joins()

    @staticmethod
    def _triple(glue) -> tuple[str, str, str]:
        if isinstance(glue, tuple):
            return glue
        return glue.number, glue.piece, glue.other

    def _units(self) -> list[GlueUnit]:
        parent = {piece: piece for piece in self.materials}

        def find(piece):
            while parent[piece] != piece:
                parent[piece] = parent[parent[piece]]
                piece = parent[piece]
            return piece

        for _, p, q in self.glues:
            if p in parent and q in parent and self.materials[p] == self.materials[q]:
                a, b = find(p), find(q)
                if a != b:
                    parent[max(a, b)] = min(a, b)
        members = collections.defaultdict(list)
        for piece in sorted(self.materials):
            members[find(piece)].append(piece)
        by_colour = collections.defaultdict(list)
        for pieces in members.values():
            by_colour[self.materials[pieces[0]]].append(pieces)
        units = []
        for material in sorted(by_colour):
            groups = sorted(by_colour[material], key=lambda row: (-len(row), row))
            units += [GlueUnit(f"{material}-{i:02d}", material, pieces) for i, pieces in enumerate(groups, 1)]
        index = {piece: unit for unit in units for piece in unit.pieces}
        for k, (_, p, q) in enumerate(self.glues):
            if p in index and index[p] is index.get(q):
                index[p].glues.append(k)
        return units

    def _joins(self) -> list[UnitJoin]:
        joins: dict[tuple[str, str], UnitJoin] = {}
        for k, (_, p, q) in enumerate(self.glues):
            a, b = self.unit_of.get(p), self.unit_of.get(q)
            if a is None or b is None or a == b:
                continue
            key = tuple(sorted((a, b)))
            joins.setdefault(key, UnitJoin(*key)).glues.append(k)
        return [joins[key] for key in sorted(joins)]

    def stage(self, piece: str, tab_piece: str) -> str:
        """`unit` for a glue inside one unit, `join` for one between units."""
        return "unit" if self.unit_of.get(piece) == self.unit_of.get(tab_piece) else "join"

    # outputs --------------------------------------------------------------------
    def unit_rows(self, status: dict | None = None) -> list[dict]:
        rows = []
        for unit in self.units:
            closes = self._closes(unit.glues, status)
            rows.append({"unit": unit.id, "material": unit.material, "pieces": len(unit.pieces),
                         "glues": len(unit.glues), "closes": closes,
                         "piece_ids": " ".join(unit.pieces)})
        return rows

    def join_rows(self, status: dict | None = None) -> list[dict]:
        unit = {u.id: u for u in self.units}
        rows = []
        for join in self.joins:
            rows.append({"unit_a": join.a, "unit_b": join.b,
                         "materials": f"{unit[join.a].material}+{unit[join.b].material}",
                         "glues": len(join.glues), "closes": self._closes(join.glues, status),
                         "numbers": " ".join(sorted((self.glues[k][0] for k in join.glues), key=_number_key))})
        return rows

    def _closes(self, glues: list[int], status: dict | None) -> str:
        if status is None:
            return ""
        return str(sum(status.get(self.glues[k], "").endswith("closes") for k in glues))

    def write(self, out: Path, status: dict | None = None) -> None:
        _csv(out / "fold3d-units.csv", self.unit_rows(status),
             ["unit", "material", "pieces", "glues", "closes", "piece_ids"])
        _csv(out / "fold3d-joins.csv", self.join_rows(status),
             ["unit_a", "unit_b", "materials", "glues", "closes", "numbers"])

    def summary(self) -> str:
        colours = collections.Counter(unit.material for unit in self.units)
        alone = sum(len(unit.pieces) == 1 for unit in self.units)
        cross = sum(len(join.glues) for join in self.joins)
        return (f"{len(self.units)} glue units (" + ", ".join(f"{n} {m}" for m, n in sorted(colours.items()))
                + f"; {alone} single pieces), {len(self.joins)} joins between units with {cross} glues")


def read_status(path: Path) -> dict | None:
    """(number, piece, tab_piece) -> status from a fold3d-glue.csv, or None when absent."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return {(r["number"], r["piece"], r["tab_piece"]): r["status"] for r in csv.DictReader(handle)}


def _number_key(number: str):
    return (0, int(number)) if number.isdigit() else (1, number)


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    from .fold3d import GlueFinder, read_pieces

    parser = argparse.ArgumentParser(description="Write the modular glue plan (units per colour, joins).")
    parser.add_argument("--config", type=Path, default=Path("data/example/example.yaml"), help="project YAML manifest")
    args = parser.parse_args()
    out = args.config.parent
    svg = out / f"{args.config.stem}.svg"
    if not svg.is_file():
        parser.error(f"{svg} not found; run make first")
    pieces = read_pieces(svg)
    glues, _ = GlueFinder(pieces).find()
    plan = GluePlan({p.id: p.material for p in pieces}, glues)
    plan.write(out, read_status(out / "fold3d-glue.csv"))
    print(plan.summary())
    print(f"Wrote {out / 'fold3d-units.csv'} and fold3d-joins.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
