"""Data for the number check page: data/<project>/check-page.json.

    python -m src.lowpoly_pdf_to_svg.fixer --config data/example/example.yaml

One card per number that did not pair up: the PDF patch around it beside the patch
around its partner (the suggested look-alike, or the copy it was paired with).
app/number-check.html opens a model folder, shows the cards, and Save writes
number-corrections.csv back into the folder; the next build applies it. The JSON
embeds scan patches, so it stays local (gitignored).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CROP_MM = 36.0        # smallest patch; grows to hold the whole line the number marks
MARGIN_MM = 10.0      # room past the line's ends, where the true corner may lie
MAX_MM = 120.0
FOLDED_MM = 0.6       # a 3D-model pair matches a card when its line length agrees this closely
SIZE_PX = 360         # patches are scaled to this so long lines do not bloat the JSON
PIECE_PX = 720        # whole-piece patches for the piece review
PIECE_MARGIN_MM = 15.0     # room past the traced outline, where a clipped piece goes on
LOOSE_MM = 4.0        # untraced line ink inside a piece beyond this much may be a missed fold


class CheckData:
    def __init__(self, folder: Path, project: str, dpi: int):
        self.folder, self.project = folder, project
        self.mm_per_px = 25.4 / dpi
        self.edges = list(csv.DictReader((folder / f"edge-numbers.csv").open(encoding="utf-8")))
        self.misreads = list(csv.DictReader((folder / f"number-misreads.csv").open(encoding="utf-8")))
        self.pages: dict[str, Image.Image] = {}
        self.lines = {(r["piece"], r["x_mm"], r["y_mm"]): [float(v) for v in r.get("line_mm", "").split()]
                      for r in self.edges}
        self.checks = {(r["piece"], r["x_mm"], r["y_mm"]): r.get("checks", "") for r in self.edges}
        self.traces = self._traces()
        self.folded = self._folded()

    def _folded(self) -> dict[frozenset, list[tuple[float, str]]]:
        """Second opinion from the folded 3D model (fold3d-glue.csv, written by the 3D
        reconstruction): pairs of an open edge and an open tab of the same length that meet once
        folded. Keyed by the two pieces; reads can differ between builds, so a pair is matched
        to a card by its pieces and line length."""
        found: dict[frozenset, list[tuple[float, str]]] = {}
        for path in (self.folder / f"fold3d-glue.csv", self.folder / "3d" / f"fold3d-glue.csv"):
            if path.is_file():
                for r in csv.DictReader(path.open(encoding="utf-8")):
                    if r["status"].startswith("proposed"):
                        found.setdefault(frozenset({r["piece"], r["tab_piece"]}), []).append((float(r["length_mm"]), r["status"]))
                break
        return found

    def _folded_status(self, m: dict) -> str:
        length = (float(m["length_mm"]) + float(m["partner_length_mm"])) / 2
        return next((status for mm, status in self.folded.get(frozenset({m["piece"], m["partner_piece"]}), [])
                     if abs(mm - length) <= FOLDED_MM), "")

    def _traces(self) -> dict[str, dict]:
        """Traced outline corners and folds per piece, in page mm, from the build cache."""
        traces = {}
        for path in sorted((self.folder / "cache").glob("page-*.pickle")):
            with path.open("rb") as handle:
                page, layout = pickle.load(handle)
            mm = page.mm_per_px
            for piece in layout.pieces:
                traces[piece.id] = {
                    "outline": [[round(x * mm, 2), round(y * mm, 2)] for x, y in piece.outline],
                    "folds": [[round(f.a[0] * mm, 2), round(f.a[1] * mm, 2), round(f.b[0] * mm, 2), round(f.b[1] * mm, 2), f.kind.value]
                              for f in piece.folds]}
        return traces

    def cards(self) -> list[dict]:
        cards, covered = [], set()
        # Hand corrections that still do not pair: the correction, its partner or the line is off.
        for r in self.edges:
            if r.get("corrected") != "yes" or r["status"] == "ok":
                continue
            copies = [r] + [o for o in self.edges if o is not r and o["number"] == r["number"]][:2]
            covered.update((o["piece"], o["x_mm"], o["y_mm"]) for o in copies)
            cards.append({"kind": "doubt", "suggested": "", "why": r["status"],
                          "panels": [self._panel(o["piece"], o["x_mm"], o["y_mm"], o["number"], o["side"], o["length_mm"])
                                     for o in copies]})
        for m in self.misreads:
            this = self._panel(m["piece"], m["x_mm"], m["y_mm"], m["read"], m["side"], m["length_mm"])
            other = self._panel(m["partner_piece"], m["partner_x_mm"], m["partner_y_mm"], m["suggested"],
                                m["partner_side"], m["partner_length_mm"])
            covered.update({(m["piece"], m["x_mm"], m["y_mm"]), (m["partner_piece"], m["partner_x_mm"], m["partner_y_mm"])})
            kind = "auto" if m.get("sure") == "applied" else "suggested"
            folded = self._folded_status(m)
            cards.append({"kind": kind, "suggested": m["suggested"], "why": m["why"], "panels": [this, other],
                          "folded": folded})
        for r in self.edges:
            if r["status"] == "ok" or (r["piece"], r["x_mm"], r["y_mm"]) in covered:
                continue
            panels = [self._panel(r["piece"], r["x_mm"], r["y_mm"], r["number"], r["side"], r["length_mm"])]
            partner = next((o for o in self.edges if o is not r and o["number"] == r["number"]
                            and o["piece"] == r["partner_piece"] and o["side"] == r["partner_side"]), None)
            if partner:
                panels.append(self._panel(partner["piece"], partner["x_mm"], partner["y_mm"], partner["number"],
                                          partner["side"], partner["length_mm"]))
            cards.append({"kind": "open", "suggested": "", "why": r["status"], "panels": panels})
        return cards

    def pieces(self) -> list[dict]:
        """One entry per traced piece for the piece review: the scan of the whole piece and how much
        its corners and folds likely need fixing (trace faults, untraced ink, numbers with no fold)."""
        faults: dict[str, list[str]] = {}
        path = self.folder / f"trace-faults.csv"
        if path.is_file():
            for r in csv.DictReader(path.open(encoding="utf-8")):
                faults.setdefault(r["piece"], []).append(f"{r['check']}: {r['detail']}")
        path = self.folder / f"fold3d-inspect.csv"   # pieces the folded 3D model doubts
        if path.is_file():
            for r in csv.DictReader(path.open(encoding="utf-8")):
                faults.setdefault(r["piece"], []).append(f"3D fold: {r['reason']}")
        suggest: dict[str, list[dict]] = {}
        path = self.folder / f"mirror-missing-folds.csv"
        if path.is_file():
            for r in csv.DictReader(path.open(encoding="utf-8")):
                if r.get("issue", "missing") != "missing":
                    continue
                suggest.setdefault(r["piece"], []).append(
                    {"kind": r["kind"], "twin": r["twin"], "a": [float(r["ax_mm"]), float(r["ay_mm"])],
                     "b": [float(r["bx_mm"]), float(r["by_mm"])]})
        twin: dict[str, str] = {}
        path = self.folder / f"mirror-mismatches.csv"
        if path.is_file():
            for r in csv.DictReader(path.open(encoding="utf-8")):
                twin[r["piece"]], twin[r["twin"]] = r["twin"], r["piece"]
        numbers: dict[str, list[dict]] = {}
        for r in self.edges:
            numbers.setdefault(r["piece"], []).append(
                {"x": r["x_mm"], "y": r["y_mm"], "read": r["number"], "status": r["status"],
                 "line": [float(v) for v in r.get("line_mm", "").split()]})
        found = []
        for piece, trace in self.traces.items():
            labels = numbers.get(piece, [])
            # Geometry only: trace faults, untraced line ink, and numbers that found no fold or line.
            lost = [l for l in labels if l["status"] in ("no tab", "line not found")]
            loose = self._loose(piece, trace, labels)
            ink_mm = loose.sum() * self.mm_per_px / 2   # ink pixels over a line about 2 px wide
            why = faults.get(piece, [])[:]
            if ink_mm > LOOSE_MM:
                why.append(f"about {ink_mm:.0f} mm of line ink inside the piece is not traced (orange): a missed fold?")
            twins = suggest.get(piece, [])
            if twins:
                why.append(f"its mirror twin {twins[0]['twin']} has {len(twins)} fold(s) missing here (orange lines)")
            if lost:
                why.append(f"{len(lost)} number(s) found no fold to sit on: a fold may be missing")
            score = 2 * len(faults.get(piece, [])) + int(ink_mm // LOOSE_MM) + 2 * len(lost) + 3 * len(twins)
            xs, ys = [q[0] for q in trace["outline"]], [q[1] for q in trace["outline"]]
            side = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * PIECE_MARGIN_MM
            image, box = self._patch(piece, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, side, PIECE_PX, loose)
            found.append({"piece": piece, "score": score, "why": why, "labels": labels, "image": image, "box": box,
                          "suggest": twins, "twin": twin.get(piece, "")})
        return found

    def _loose(self, piece: str, trace: dict, labels: list[dict]) -> np.ndarray:
        """Page mask of line ink inside the piece that no traced fold, cut or number explains."""
        gray = np.asarray(self._page(piece))
        px = lambda pts: np.round(np.asarray(pts, float) / self.mm_per_px).astype(np.int32)
        reach = round(1.0 / self.mm_per_px)
        inside = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(inside, [px(trace["outline"])], 1)
        cv2.polylines(inside, [px(trace["outline"])], True, 0, 2 * reach + 1)
        known = np.zeros(gray.shape, np.uint8)
        for f in trace["folds"]:
            cv2.line(known, *[tuple(int(v) for v in q) for q in px([f[:2], f[2:]])], 1, 2 * reach + 1)
        for l in labels:
            cv2.circle(known, tuple(int(v) for v in px([float(l["x"]), float(l["y"])])), round(3.5 / self.mm_per_px), 1, -1)
        loose = (gray < 215) & (inside > 0) & (known == 0)
        # Specks (scan noise, a stray dot) are no line.
        count, parts, stats, _ = cv2.connectedComponentsWithStats(loose.astype(np.uint8), connectivity=8)
        keep = np.zeros(count, bool)
        keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= 0.8 / self.mm_per_px
        return keep[parts]

    def _page(self, piece: str) -> Image.Image:
        page = piece.split("-")[1]
        if page not in self.pages:
            found = sorted((self.folder / "previews").glob(f"page-{page}-*.png"))
            self.pages[page] = Image.open(found[0]).convert("L") if found else Image.new("L", (10, 10), 255)
        return self.pages[page]

    def _panel(self, piece: str, x: str, y: str, number: str, side: str, length: str) -> dict:
        line = self.lines.get((piece, x, y), [])
        image, box = self._crop(piece, float(x), float(y), line)
        return {"piece": piece, "x": x, "y": y, "read": number, "side": side, "length": length,
                "image": image, "box": box, "line": line, "checks": self.checks.get((piece, x, y), "")}

    def _crop(self, piece: str, x_mm: float, y_mm: float, line: list[float]) -> tuple[str, list[float]]:
        """Plain scan patch around the number, wide enough to show the whole line it marks,
        and its box on the page [left, top, side] in mm; the page draws the trace on top."""
        xs, ys = [x_mm] + line[0::2], [y_mm] + line[1::2]
        side = min(MAX_MM, max(CROP_MM, max(xs) - min(xs) + 2 * MARGIN_MM, max(ys) - min(ys) + 2 * MARGIN_MM))
        return self._patch(piece, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, side, SIZE_PX)

    def _patch(self, piece: str, cx: float, cy: float, side: float, size: int,
               mark: np.ndarray | None = None) -> tuple[str, list[float]]:
        """Scan patch [left, top, side] mm, scaled to size px; ink under `mark` is tinted orange."""
        image = self._page(piece)
        if mark is not None and mark.any():
            rgb = np.asarray(image.convert("RGB")).copy()
            rgb[cv2.dilate(mark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (255, 120, 0)
            image = Image.fromarray(rgb)
        left, top = cx - side / 2, cy - side / 2
        px = lambda mm: round(mm / self.mm_per_px)
        crop = Image.new(image.mode, (px(side), px(side)), "white")   # white past the page edge
        crop.paste(image, (-px(left), -px(top)))
        crop = crop.resize((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, "JPEG", quality=80)
        box = [round(px(left) * self.mm_per_px, 3), round(px(top) * self.mm_per_px, 3), round(px(side) * self.mm_per_px, 3)]
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), box

    def write(self, path: Path) -> int:
        cards = self.cards()
        path.write_text(json.dumps({"project": self.project, "cards": cards, "traces": self.traces,
                                    "pieces": self.pieces()}), encoding="utf-8")
        return len(cards)




def main() -> int:
    parser = argparse.ArgumentParser(description="Write the number check data for app/number-check.html.")
    parser.add_argument("--config", type=Path, default=Path("data/example/example.yaml"), help="project YAML manifest")
    parser.add_argument("--dpi", type=int, default=180, help="resolution the previews were rendered at")
    args = parser.parse_args()
    folder, project = args.config.parent, args.config.stem
    for name, pattern in (("previews", "*.png"), ("cache", "*.pickle")):
        if not any((folder / name).glob(pattern)):
            parser.error(f"{folder / name} is empty or missing: run make first (it is not committed)")
    try:
        count = CheckData(folder, project, args.dpi).write(folder / f"check-page.json")
    except OSError as error:
        parser.error(str(error))
    print(f"Wrote {folder / 'check-page.json'} with {count} numbers to check; "
          "open app/number-check.html and pick the folder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
