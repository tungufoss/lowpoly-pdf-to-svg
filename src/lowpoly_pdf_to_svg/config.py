"""Project manifest: which PDF pages are printed on which cardstock."""

from __future__ import annotations

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

    @classmethod
    def load(cls, path: Path) -> "Project":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            source, output = raw["source"], raw["output"]
            materials = tuple(
                Material(str(m["id"]), str(m["name"]), tuple(int(p) for p in m.get("pages") or ()),
                         tuple(str(p) for p in m.get("pieces") or ()))
                for m in raw["materials"]
            )
            project = cls(
                name=str(raw["project"]),
                pdf=(path.parent / source["pdf"]).resolve(),
                page_size=str(source["page_size"]).lower(),
                sheet_size=str(output["default_sheet"]).lower(),
                materials=materials,
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
        return seen

    def pages(self) -> list[tuple[int, Material]]:
        """Every assigned page with its material, rejecting pages listed twice."""
        seen: dict[int, Material] = {}
        for material in self.materials:
            for page in material.pages:
                if page in seen:
                    raise ValueError(f"page {page} is assigned to both {seen[page].id} and {material.id}")
                seen[page] = material
        return sorted(seen.items())
