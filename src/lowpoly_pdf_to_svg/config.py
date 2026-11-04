"""Project manifest: which PDF pages are printed on which cardstock.

Colours (lowpoly #88): an optional YAML `palette:` names the cardstock colours at hand
(name -> hex). A material's `color:` is a hex or a palette name. Pieces are recoloured either in
the YAML (`pieces:` under a material) or in recolour.csv beside the YAML (piece,material), which
the 3D viewer writes; recolour.csv wins. A palette name with no material of its own becomes one
when a piece is recoloured to it. A recolour needs no retrace: re-export and re-nest.
"""

from __future__ import annotations

import csv
import re

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .model import FoldKind

# Sheet sizes in millimetres (width, height).
SHEETS_MM = {
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
    "12x12": (304.8, 304.8),
}


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    pages: tuple[int, ...]
    pieces: tuple[str, ...] = ()   # pieces cut from this cardstock although their page is printed for another
    color: str = ""                # fill colour (hex), from `color:` or the palette; "" = default for the id


@dataclass(frozen=True)
class Project:
    name: str
    pdf: Path
    page_size: str
    sheet_size: str
    materials: tuple[Material, ...]
    # How the template draws each fold. Pepakura's default: dash-dot = mountain, dashed = valley.
    folds: dict[str, FoldKind] = field(default_factory=lambda: {"dash_dot": FoldKind.MOUNTAIN,
                                                                "dashed": FoldKind.VALLEY})
    palette: dict[str, str] = field(default_factory=dict)       # colour name -> hex
    recolour: dict[str, str] = field(default_factory=dict)      # piece -> material id (recolour.csv)

    @classmethod
    def load(cls, path: Path) -> "Project":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            source, output = raw["source"], raw["output"]
            palette = {str(k): _hex(v, f"{path}: palette {k}") for k, v in (raw.get("palette") or {}).items()}
            colour = lambda m: (palette.get(str(m["color"])) or _hex(m["color"], f"{path}: material {m['id']} color")
                                if m.get("color") else palette.get(str(m["id"]), ""))
            materials = tuple(
                Material(str(m["id"]), str(m["name"]), tuple(int(p) for p in m.get("pages") or ()),
                         tuple(str(p) for p in m.get("pieces") or ()), colour(m))
                for m in raw["materials"]
            )
            recolour = read_recolour(path.parent / RECOLOUR_FILE)
            known = {m.id for m in materials}
            for name in sorted(set(recolour.values()) - known):
                if name not in palette:
                    raise ValueError(f"{path.parent / RECOLOUR_FILE}: {name} is neither a material nor in the palette")
                materials += (Material(name, f"{name.replace('_', ' ').capitalize()} cardstock", (), (), palette[name]),)
            project = cls(
                name=str(raw["project"]),
                pdf=(path.parent / source["pdf"]).resolve(),
                page_size=str(source["page_size"]).lower(),
                sheet_size=str(output["default_sheet"]).lower(),
                materials=materials,
                palette=palette,
                recolour=recolour,
            )
            if "folds" in raw:
                folds = {str(style): FoldKind(str(kind).lower()) for style, kind in raw["folds"].items()}
                if set(folds) != {"dash_dot", "dashed"}:
                    raise ValueError(f"{path}: folds must name both dash_dot and dashed")
                object.__setattr__(project, "folds", folds)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}: missing or malformed field {error}") from error
        if not materials:
            raise ValueError(f"{path}: no materials were found")
        if project.sheet_size not in SHEETS_MM:
            raise ValueError(f"{path}: unknown sheet {project.sheet_size!r}; use one of {sorted(SHEETS_MM)}")
        return project

    def piece_materials(self) -> dict[str, Material]:
        """Pieces moved onto another cardstock than their page's, rejecting pieces listed twice."""
        seen: dict[str, Material] = {}
        for material in self.materials:
            for piece in material.pieces:
                if piece in seen:
                    raise ValueError(f"{piece} is assigned to both {seen[piece].id} and {material.id}")
                seen[piece] = material
        by_id = {m.id: m for m in self.materials}
        seen.update({piece: by_id[material] for piece, material in self.recolour.items()})
        return seen

    def colours(self) -> dict[str, str]:
        """Hex fill per material id, plus every palette colour (for the viewer's recolour menu)."""
        return {**self.palette, **{m.id: m.color for m in self.materials if m.color}}

    def pages(self) -> list[tuple[int, Material]]:
        """Every assigned page with its material, rejecting pages listed twice."""
        seen: dict[int, Material] = {}
        for material in self.materials:
            for page in material.pages:
                if page in seen:
                    raise ValueError(f"page {page} is assigned to both {seen[page].id} and {material.id}")
                seen[page] = material
        return sorted(seen.items())


RECOLOUR_FILE = "recolour.csv"


def read_recolour(path: Path) -> dict[str, str]:
    """piece -> material id from recolour.csv (written by the 3D viewer's Recolour mode)."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["piece"].strip(): row["material"].strip() for row in csv.DictReader(handle)
                if (row.get("piece") or "").strip() and (row.get("material") or "").strip()}


def _hex(value, where: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        raise ValueError(f"{where}: {text!r} is not a #rrggbb colour")
    return text.lower()
