"""Recolour pieces or single faces: the YAML palette for the 3D viewer, and the viewer's recolour back into the YAML.

A piece's cardstock is its page's material unless a material lists it under `pieces:`, and a face's
is its piece's unless a material lists it under `faces:`. The viewer (app/fold3d-viewer.html,
Recolour) moves pieces or single faces between palette colours and saves recolour.csv
(piece,material; a row names a piece, piece-04-06, or one face, piece-04-06/5). This module writes
that into the YAML's `materials:` `pieces:` and `faces:` lists, keeping the rest of the file and its
comments, so a recolour is a YAML edit plus `make export` and a re-nest, never a hand edit of pages.
A piece with faces of two colours is cut apart along the colour boundary by `make segment`, which
reads the colours from the YAML (`yaml_colours`).

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
    """Which cardstock each piece, and each single face, is cut from, before and after a list of
    changes. A change names a piece (piece-04-06: the whole piece) or a face (piece-04-06/5)."""

    def __init__(self, project: Project):
        self.project = project
        self.page_material = {page: m.id for page, m in project.pages()}
        self.moved = {piece: m.id for piece, m in project.piece_materials().items()}
        self.face_moved = {face: m.id for face, m in project.face_materials().items()}
        self.ids = [m.id for m in project.materials] + [m.id for m in project.palette]

    def material(self, key: str) -> str:
        """A piece's material, or a face's (its own if the YAML gives one, else its piece's)."""
        piece = key.split("/")[0]
        own = self.face_moved.get(key) if "/" in key else None
        return own or self.moved.get(piece) or self.page_material.get(piece_page(piece) or -1, "")

    def apply(self, changes: dict[str, str]) -> dict[str, dict[str, list[str]]]:
        """The `pieces:` and `faces:` lists of every material after the changes. Whole pieces first:
        a piece row also clears that piece's face rows. A piece back on its page's colour, or a face
        back on its piece's, leaves the lists."""
        unknown = sorted(set(changes.values()) - set(self.ids))
        if unknown:
            raise ValueError(f"not in materials or palette: {', '.join(unknown)}")
        moved, face_moved = dict(self.moved), dict(self.face_moved)
        for piece, material in changes.items():
            if "/" in piece:
                continue
            if self.page_material.get(piece_page(piece) or -1) == material:
                moved.pop(piece, None)
            else:
                moved[piece] = material
            for face in [f for f in face_moved if f.split("/")[0] == piece]:
                del face_moved[face]
        for face, material in changes.items():
            if "/" not in face:
                continue
            piece = face.split("/")[0]
            if (moved.get(piece) or self.page_material.get(piece_page(piece) or -1)) == material:
                face_moved.pop(face, None)
            else:
                face_moved[face] = material
        lists = {i: {"pieces": [], "faces": []} for i in self.ids}
        for piece in sorted(moved):
            lists[moved[piece]]["pieces"].append(piece)
        for face in sorted(face_moved, key=_face_key):
            lists[face_moved[face]]["faces"].append(face)
        return lists

    def split_pieces(self, lists: dict[str, dict[str, list[str]]]) -> list[str]:
        """Pieces with faces of another colour: they are cut in two or more at the re-cut."""
        return sorted({face.split("/")[0] for entry in lists.values() for face in entry["faces"]})

    @staticmethod
    def read_csv(path: Path) -> dict[str, str]:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["piece"].strip(): row["material"].strip() for row in csv.DictReader(handle)
                    if row.get("piece") and row.get("material")}


def _face_key(face: str):
    piece, _, k = face.partition("/")
    return piece, int(k) if k.isdigit() else k


class YamlMaterials:
    """Rewrites only the `pieces:` and `faces:` entries in the YAML `materials:` list, line by line,
    so the comments and layout of everything else stay as Helga wrote them."""

    ITEM = re.compile(r"^(\s*)- id:\s*(\S+)")
    KEYS = ("pieces", "faces")

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
    def _list_text(indent: str, key: str, names: list[str]) -> list[str]:
        if not names:
            return []
        lead = f"{indent}  {key}: ["
        groups: dict = {}
        for name in names:   # one line per page (pieces) or per piece (faces)
            groups.setdefault(name.split("/")[0] if key == "faces" else piece_page(name), []).append(name)
        rows = [", ".join(group) for group in groups.values()]
        pad = " " * len(lead)
        return [(lead if k == 0 else pad) + row + ("]\n" if k == len(rows) - 1 else ",\n")
                for k, row in enumerate(rows)]

    def write(self, lists: dict[str, dict[str, list[str]]], spare: dict[str, Material],
              before: dict[str, dict[str, list[str]]] | None = None) -> str:
        """`before`: the lists as the YAML has them now; a list that is unchanged keeps its lines
        (and their comments) as they are."""
        before = before or {}
        empty = {key: [] for key in self.KEYS}
        start, end = self._block()
        items = self._items(start, end)
        seen = {name for name, *_ in items}
        out = self.lines[:start + 1]
        cursor = start + 1
        for name, head, stop, indent in items:
            out += self.lines[cursor:head]
            body = self.lines[head:stop]
            was, now = before.get(name, empty), lists.get(name, empty)
            changed = [key for key in self.KEYS if sorted(was.get(key, [])) != sorted(now.get(key, []))]
            if not changed:
                out += body
                cursor = stop
                continue
            key = re.compile(rf"^{re.escape(indent)}  (\w+):")
            keep, k = [], 0
            while k < len(body):
                match = key.match(body[k])
                if match and match.group(1) in changed:
                    k += 1
                    while k < len(body) and not key.match(body[k]) and body[k].startswith(indent + "   "):
                        k += 1
                    continue
                keep.append(body[k])
                k += 1
            # The new lists: after the last key line of the item (before trailing blanks/comments).
            last = max((j for j, l in enumerate(keep) if j == 0 or key.match(l) or l.startswith(indent + "   ")),
                       default=0)
            added = [line for k_ in changed for line in self._list_text(indent, k_, now.get(k_, []))]
            out += keep[:last + 1] + added + keep[last + 1:]
            cursor = stop
        out += self.lines[cursor:end]
        indent = items[0][3] if items else "  "
        for name, entry in lists.items():
            if name in seen or not any(entry.values()):
                continue
            material = spare[name]
            out += [f"{indent}- id: {name}\n", f"{indent}  name: {material.name}\n",
                    f"{indent}  color: \"{material.color}\"\n"]
            out += [line for k_ in self.KEYS for line in self._list_text(indent, k_, entry.get(k_, []))]
        return "".join(out + self.lines[end:])


def yaml_colours(data: dict, fold3d: Path) -> int:
    """Colour fold3d.json's pieces and faces as the YAML next to it says now (recolours since the
    fold included), so a re-cut keeps faces of different cardstock apart. Returns the faces changed."""
    config = next(iter(sorted(fold3d.parent.glob("*.yaml"))), None)
    if config is None:
        return 0
    try:
        recolour = Recolour(Project.load(config))
    except ValueError:
        return 0
    changed = 0
    for piece, entry in data["pieces"].items():
        entry["material"] = recolour.material(piece) or entry.get("material", "")
    faces = data.setdefault("face_material", {})
    for face in data["faces"]:
        material = recolour.material(face["id"]) or data["pieces"].get(face["piece"], {}).get("material", "")
        if material != data["pieces"].get(face["piece"], {}).get("material", ""):
            faces[face["id"]] = material
            changed += 1
    return changed


def palette_json(project: Project) -> dict:
    """What the viewer needs to colour pieces as the YAML says, without a new fold."""
    width, height = SHEETS_MM[project.sheet_size]
    entry = lambda m, spare: {"id": m.id, "name": m.name, "color": m.color, "pages": list(m.pages),
                              "pieces": list(m.pieces), "faces": list(m.faces), "spare": spare}
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
        before = {m.id: {"pieces": list(m.pieces), "faces": list(m.faces)} for m in project.materials}
        text = YamlMaterials(args.config.read_text(encoding="utf-8")).write(
            lists, {m.id: m for m in project.palette}, before)
        args.config.write_text(text, encoding="utf-8")
        project = Project.load(args.config)   # check the rewrite reads back as asked
        check = Recolour(project)
        wrong = sorted(p for p, m in changes.items() if check.material(p) != m)
        if wrong:
            raise SystemExit(f"recolour did not take for {', '.join(wrong)}")
        print(f"{args.config}: {len(moved)} pieces or faces recoloured"
              + (" (" + ", ".join(f"{p} -> {m}" for p, m in sorted(moved.items())) + ")" if moved else ""))
        split = check.split_pieces(check.apply({}))
        if split:
            print(f"split by colour ({len(split)}): {', '.join(split)}\n"
                  f"  the traced export keeps these whole in their piece's colour; cut them apart with\n"
                  f"  make segment PIECES={','.join(split)}  (faces of different colour never share a piece)")
    path = args.config.parent / "palette.json"
    path.write_text(json.dumps(palette_json(project), indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
