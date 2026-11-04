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

# Cardstock colours used when neither the YAML material nor its `palette:` entry gives a `color:`.
DEFAULT_COLORS = {"red": "#c62828", "green": "#2e7d32", "yellow": "#f9d71c", "black": "#212121",
                  "white": "#ffffff", "beige": "#e8d9b5", "blue": "#1565c0", "orange": "#ef6c00",
                  "brown": "#6d4c41", "grey": "#9e9e9e", "gray": "#9e9e9e", "pink": "#f48fb1",
                  "purple": "#6a1b9a", "rust": "#b7410e", "gold": "#d4a017"}
FALLBACK_COLOR = "#cccccc"


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    pages: tuple[int, ...]
    pieces: tuple[str, ...] = ()   # pieces cut from this cardstock although their page is printed for another
    color: str = ""                # fill colour (#rrggbb); empty: the palette's, else DEFAULT_COLORS
    faces: tuple[str, ...] = ()    # single faces (piece-04-06/5) of this colour inside a piece of another:
                                   # the piece is split along the colour boundary when it is re-cut (make segment)


class Palette:
    """The cardstock colours a model may use: every material, plus the YAML `palette:` entries
    (cardstock on hand that no page uses yet). A recolour moves pieces between these ids."""

    def __init__(self, raw: dict):
        entries = raw.get("palette") or []
        if isinstance(entries, dict):   # palette: {blue: {name: ..., color: ...}} also reads
            entries = [{"id": k, **(v or {})} for k, v in entries.items()]
        self.extra = {str(e["id"]): (str(e.get("name") or e["id"]), str(e.get("color") or "")) for e in entries}

    def name(self, material_id: str, given: str = "") -> str:
        return given or self.extra.get(material_id, ("", ""))[0] or material_id

    def color(self, material_id: str, given: str = "") -> str:
        return (given or self.extra.get(material_id, ("", ""))[1]
                or DEFAULT_COLORS.get(material_id.lower(), FALLBACK_COLOR))


@dataclass(frozen=True)
class Project:
    name: str
    pdf: Path
    page_size: str
    sheet_size: str
    materials: tuple[Material, ...]
    palette: tuple[Material, ...] = ()   # spare cardstock from `palette:` that no piece uses yet
    # How the template draws each fold. Pepakura's default: dash-dot = mountain, dashed = valley.
    folds: dict[str, FoldKind] = field(default_factory=lambda: {"dash_dot": FoldKind.MOUNTAIN,
                                                                "dashed": FoldKind.VALLEY})

    @classmethod
    def load(cls, path: Path) -> "Project":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            source, output = raw["source"], raw["output"]
            palette = Palette(raw)
            materials = tuple(
                Material(str(m["id"]), palette.name(str(m["id"]), str(m.get("name") or "")),
                         tuple(int(p) for p in m.get("pages") or ()),
                         tuple(str(p) for p in m.get("pieces") or ()),
                         palette.color(str(m["id"]), str(m.get("color") or "")),
                         tuple(str(f) for f in m.get("faces") or ()))
                for m in raw["materials"]
            )
            used = {m.id for m in materials}
            spare = tuple(Material(i, palette.name(i), (), (), palette.color(i)) for i in palette.extra if i not in used)
            project = cls(
                name=str(raw["project"]),
                pdf=(path.parent / source["pdf"]).resolve(),
                page_size=str(source["page_size"]).lower(),
                sheet_size=str(output["default_sheet"]).lower(),
                materials=materials,
                palette=spare,
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

    def face_materials(self) -> dict[str, Material]:
        """Single faces given another colour than their piece's, rejecting faces listed twice."""
        seen: dict[str, Material] = {}
        for material in self.materials:
            for face in material.faces:
                if face in seen:
                    raise ValueError(f"{face} is assigned to both {seen[face].id} and {material.id}")
                seen[face] = material
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
