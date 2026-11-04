"""Recolour pieces: the YAML palette for the 3D viewer, and the viewer's recolour back into the YAML.

A piece's cardstock is its page's material unless a material lists it under `pieces:`. The viewer
(app/fold3d-viewer.html, Recolour) moves pieces between palette colours and saves recolour.csv
(piece,material). This module writes that into the YAML's `materials:` `pieces:` lists, keeping
the rest of the file and its comments, so a recolour is a YAML edit plus `make export` and a
re-nest, never a hand edit of pages.

    python -m src.lowpoly_pdf_to_svg.palette --config data/<p>/<p>.yaml            # palette.json
    python -m src.lowpoly_pdf_to_svg.palette --config data/<p>/<p>.yaml --apply data/<p>/recolour.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .config import SHEETS_MM, Material, Project

PIECE_PAGE = re.compile(r"piece-(\d+)-")


def piece_page(piece: str) -> int | None:
    match = PIECE_PAGE.match(piece)
    return int(match.group(1)) if match else None


class Recolour:
    """Which cardstock each piece is cut from, before and after a list of changes."""

    def __init__(self, project: Project):
        self.project = project
        self.page_material = {page: m.id for page, m in project.pages()}
        self.moved = {piece: m.id for piece, m in project.piece_materials().items()}
        self.ids = [m.id for m in project.materials] + [m.id for m in project.palette]

    def material(self, piece: str) -> str:
        return self.moved.get(piece) or self.page_material.get(piece_page(piece) or -1, "")

    def apply(self, changes: dict[str, str]) -> dict[str, list[str]]:
        """The `pieces:` list of every material after the changes; a piece back on its page's
        colour leaves the lists."""
        unknown = sorted(set(changes.values()) - set(self.ids))
        if unknown:
            raise ValueError(f"not in materials or palette: {', '.join(unknown)}")
        moved = dict(self.moved)
        for piece, material in changes.items():
            if self.page_material.get(piece_page(piece) or -1) == material:
                moved.pop(piece, None)
            else:
                moved[piece] = material
        lists: dict[str, list[str]] = {i: [] for i in self.ids}
        for piece in sorted(moved):
            lists[moved[piece]].append(piece)
        return lists

    @staticmethod
    def read_csv(path: Path) -> dict[str, str]:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["piece"].strip(): row["material"].strip() for row in csv.DictReader(handle)
                    if row.get("piece") and row.get("material")}


class YamlMaterials:
    """Rewrites only the `pieces:` entries in the YAML `materials:` list, line by line, so the
    comments and layout of everything else stay as Helga wrote them."""

    ITEM = re.compile(r"^(\s*)- id:\s*(\S+)")

    def __init__(self, text: str):
        self.lines = text.splitlines(keepends=True)

    def _block(self) -> tuple[int, int]:
        start = next((i for i, l in enumerate(self.lines) if re.match(r"^materials:\s*(#.*)?$", l)), None)
        if start is None:
            raise ValueError("the YAML has no top-level materials: list")
        end = next((i for i in range(start + 1, len(self.lines)) if re.match(r"^[A-Za-z_]", self.lines[i])),
                   len(self.lines))
        while end > start + 1 and re.match(r"^(#.*)?\s*$", self.lines[end - 1]):  # comments of the next key
            end -= 1
        return start, end

    def _items(self, start: int, end: int) -> list[tuple[str, int, int, str]]:
        heads = [(i, m) for i in range(start + 1, end) if (m := self.ITEM.match(self.lines[i]))]
        return [(m.group(2).strip("'\""), i, heads[k + 1][0] if k + 1 < len(heads) else end, m.group(1))
                for k, (i, m) in enumerate(heads)]

    @staticmethod
    def _pieces_text(indent: str, pieces: list[str]) -> list[str]:
        if not pieces:
            return []
        lead = f"{indent}  pieces: ["
        by_page: dict[int | None, list[str]] = {}
        for piece in pieces:
            by_page.setdefault(piece_page(piece), []).append(piece)
        rows = [", ".join(group) for group in by_page.values()]
        pad = " " * len(lead)
        return [(lead if k == 0 else pad) + row + ("]\n" if k == len(rows) - 1 else ",\n")
                for k, row in enumerate(rows)]

    def write(self, lists: dict[str, list[str]], spare: dict[str, Material],
              before: dict[str, list[str]] | None = None) -> str:
        """`before`: the lists as the YAML has them now; a material whose list is unchanged keeps
        its lines (and their comments) as they are."""
        before = before or {}
        start, end = self._block()
        items = self._items(start, end)
        seen = {name for name, *_ in items}
        out = self.lines[:start + 1]
        cursor = start + 1
        for name, head, stop, indent in items:
            out += self.lines[cursor:head]
            body = self.lines[head:stop]
            if name in before and sorted(before[name]) == sorted(lists.get(name, [])):
                out += body
                cursor = stop
                continue
            key = re.compile(rf"^{re.escape(indent)}  (\w+):")
            keep, k = [], 0
            while k < len(body):
                match = key.match(body[k])
                if match and match.group(1) == "pieces":
                    k += 1
                    while k < len(body) and not key.match(body[k]) and body[k].startswith(indent + "   "):
                        k += 1
                    continue
                keep.append(body[k])
                k += 1
            # The new pieces: after the last key line of the item (before trailing blanks/comments).
            last = max((j for j, l in enumerate(keep) if j == 0 or key.match(l) or l.startswith(indent + "   ")),
                       default=0)
            out += keep[:last + 1] + self._pieces_text(indent, lists.get(name, [])) + keep[last + 1:]
            cursor = stop
        out += self.lines[cursor:end]
        indent = items[0][3] if items else "  "
        for name, pieces in lists.items():
            if name in seen or not pieces:
                continue
            material = spare[name]
            out += [f"{indent}- id: {name}\n", f"{indent}  name: {material.name}\n",
                    f"{indent}  color: \"{material.color}\"\n"] + self._pieces_text(indent, pieces)
        return "".join(out + self.lines[end:])


def palette_json(project: Project) -> dict:
    """What the viewer needs to colour pieces as the YAML says, without a new fold."""
    width, height = SHEETS_MM[project.sheet_size]
    entry = lambda m, spare: {"id": m.id, "name": m.name, "color": m.color, "pages": list(m.pages),
                              "pieces": list(m.pieces), "spare": spare}
    return {"project": project.name, "sheet": project.sheet_size, "sheet_mm": [width, height],
            "materials": [entry(m, False) for m in project.materials] + [entry(m, True) for m in project.palette]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Palette for the viewer, and recolours back into the YAML.")
    parser.add_argument("--config", type=Path, required=True, help="project YAML manifest")
    parser.add_argument("--apply", type=Path, help="recolour.csv (piece,material) from the viewer")
    args = parser.parse_args()
    project = Project.load(args.config)
    if args.apply:
        changes = Recolour.read_csv(args.apply)
        recolour = Recolour(project)
        moved = {p: m for p, m in changes.items() if recolour.material(p) != m}
        lists = recolour.apply(changes)
        text = YamlMaterials(args.config.read_text(encoding="utf-8")).write(
            lists, {m.id: m for m in project.palette}, {m.id: list(m.pieces) for m in project.materials})
        args.config.write_text(text, encoding="utf-8")
        project = Project.load(args.config)   # check the rewrite reads back as asked
        check = Recolour(project)
        wrong = sorted(p for p, m in changes.items() if check.material(p) != m)
        if wrong:
            raise SystemExit(f"recolour did not take for {', '.join(wrong)}")
        print(f"{args.config}: {len(moved)} pieces recoloured"
              + (" (" + ", ".join(f"{p} -> {m}" for p, m in sorted(moved.items())) + ")" if moved else ""))
    path = args.config.parent / "palette.json"
    path.write_text(json.dumps(palette_json(project), indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
