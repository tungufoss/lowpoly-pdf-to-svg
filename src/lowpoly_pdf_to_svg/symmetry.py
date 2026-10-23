"""Stage 2d: use the model's left-right symmetry to check numbers and traced lines.

Most pieces have a mirror twin, cut from the same cardstock. Twins are found by their
numbered lines: walking round one piece, the lengths of its marked lines (cut edges and
tab folds) read the same as its twin's walked the other way, with gaps where a number
found no line. An exact mirror also matches outline for outline, tabs included; a mirror
variation has the same body with its glue tabs on other edges. That pairs every numbered line with its
mirror line, which gives two checks:

* a line and its mirror line should be the same length; when not, one was traced off;
* folds mirror too: a fold on one twin with nothing at its mirror place on the other is
  likely a fold the trace missed there (fold_gaps);
* glue pairs mirror too. If line a on piece A glues to piece C, then a's mirror line on
  A's twin glues to C's twin, at the mirror of C's line. So the number on that line
  is predicted from the clean pair on the other side.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .model import Label, PageLayout

LENGTH_MM = 1.0       # mirror lines agree within this
NEAR_MM = 3.0         # a line this close to its mirror line is still the same line, traced off
MIN_SHARE = 0.7       # share of lines that must agree for two pieces to count as twins
EXACT_MM = 1.0        # outlines this close (RMS, tabs included) make an exact mirror
SAMPLES = 96          # outline points compared for exact mirrors
FIT_MM = 1.5          # twins whose matched lines map onto each other this closely (RMS) can compare folds
FOLD_MM = 2.0         # a mirrored fold whose ends land this close to a twin's fold is the same fold
ON_CUT_MM = 1.0       # a mirrored fold this close to the twin's cut is a glue tab moved elsewhere
MOVE_MIN_MM = 0.2     # mirror corners closer than this are left as traced
MOVE_MAX_MM = 1.5     # corners further apart are not one corner (a tab placed elsewhere)
COPY_MAX_MM = 3.0     # ... unless one twin folds cleanly in 3D: then it is the truth up to this far


@dataclass
class Twin:
    a: str
    b: str
    lines: list[tuple[Label, Label]]   # numbered lines of a with their mirror lines on b
    off: int                           # paired lines whose lengths disagree
    exact: bool                        # whole outline mirrors, tabs included; else tabs differ


class Symmetry:
    def __init__(self, layouts: list[tuple[PageLayout, float]]):
        self.mm, self.material, self.outline, self.folds = {}, {}, {}, {}
        self.lines: dict[str, list[Label]] = {}
        for layout, mm in layouts:
            for piece in layout.pieces:
                outline = np.array(piece.outline, float)
                marked = [l for l in piece.labels if l.edge is not None]
                marked.sort(key=lambda l: _along(outline, np.mean(l.edge, axis=0)))
                self.lines[piece.id], self.mm[piece.id] = marked, mm
                self.material[piece.id] = piece.material
                self.outline[piece.id] = outline * mm
                self.folds[piece.id] = [(np.array(f.a) * mm, np.array(f.b) * mm, f.kind.value) for f in piece.folds]
        self.twins = self._pair()
        self.twin_of = {t.a: t.b for t in self.twins} | {t.b: t.a for t in self.twins}
        self.mirror_line = {id(x): (t.b, y) for t in self.twins for x, y in t.lines}
        self.mirror_line |= {id(y): (t.a, x) for t in self.twins for x, y in t.lines}

    def _lengths(self, piece: str) -> np.ndarray:
        return np.array([l.edge_length * self.mm[piece] for l in self.lines[piece]])

    def _align(self, a: str, b: str) -> tuple[float, list[tuple[int, int]]] | None:
        """Walk a forwards and b backwards round the outline and line up their numbered
        lines, allowing gaps where a number found no line. Returns (share of lines that
        agree, index pairs) for the best starting point."""
        la, lb = self._lengths(a), self._lengths(b)[::-1]
        if min(len(la), len(lb)) < 3:
            return None
        best = None
        for k in range(len(lb)):
            score, pairs = _sequence(la, np.roll(lb, k))
            agree = sum(abs(la[i] - np.roll(lb, k)[j]) <= LENGTH_MM for i, j in pairs)
            share = agree / max(len(la), len(lb))
            if best is None or (share, score) > best[:2]:
                best = (share, score, [(i, (len(lb) - 1 - (j - k) % len(lb))) for i, j in pairs])
        return best[0], best[2]

    def _exact(self, a: str, b: str) -> bool:
        """Do the outlines, tabs included, mirror each other?"""
        pa, pb = _resample(self.outline[a]), _resample(self.outline[b])[::-1] * [-1, 1]
        for k in range(SAMPLES):
            u, _, vt = np.linalg.svd(np.roll(pb, k, axis=0).T @ pa)
            rotation = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
            if np.sqrt(((np.roll(pb, k, axis=0) @ rotation - pa) ** 2).sum(1).mean()) <= EXACT_MM:
                return True
        return False

    def _pair(self) -> list[Twin]:
        ids = sorted(self.lines)
        size = {i: _perimeter(self.outline[i]) for i in ids}
        found = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                # A twin is cut from the same cardstock and is about the same size.
                if self.material[a] != self.material[b] or abs(size[a] - size[b]) > 0.25 * size[a]:
                    continue
                fit = self._align(a, b)
                if fit and fit[0] >= MIN_SHARE:
                    found.append((fit[0], a, b, fit[1]))
        twins, taken = [], set()
        for share, a, b, pairs in sorted(found, key=lambda f: -f[0]):
            if a in taken or b in taken:
                continue
            taken |= {a, b}
            lines = [(self.lines[a][i], self.lines[b][j]) for i, j in pairs]
            off = sum(abs(x.edge_length * self.mm[a] - y.edge_length * self.mm[b]) > LENGTH_MM for x, y in lines)
            twins.append(Twin(a, b, lines, off, self._exact(a, b)))
        return twins

    def predictions(self, records: list[dict]) -> list[dict]:
        """Numbers that do not pair up, with the number their twin's clean pair predicts."""
        by_label = {id(r["_row"].label): r for r in records}
        by_number: dict[str, list[dict]] = {}
        for r in records:
            by_number.setdefault(r["number"], []).append(r)
        found = []
        for r in records:
            if r["status"] == "ok" or id(r["_row"].label) not in self.mirror_line:
                continue
            twin, mirror = self.mirror_line[id(r["_row"].label)]
            source = by_label.get(id(mirror))
            # The twin's line must be cleanly paired with a piece that has its own twin.
            if not source or source["status"] != "ok" or source["partner_piece"] not in self.twin_of:
                continue
            partner = next((o for o in by_number[source["number"]] if o is not source and o["piece"] == source["partner_piece"]), None)
            if not partner or id(partner["_row"].label) not in self.mirror_line:
                continue
            target_piece, target = self.mirror_line[id(partner["_row"].label)]
            if target_piece == source["piece"] or target is r["_row"].label:
                continue
            expected = by_label.get(id(target))
            if not expected or expected["number"] == r["number"]:
                continue
            # Sure when the predicted number is the only other copy of what the target line says,
            # sits on the other side, and its line has this line's length.
            number = expected["number"]
            copies = by_number.get(number, [])
            sure = (len(copies) == 1 and copies[0] is expected and expected["side"] in ("edge", "tab")
                    and r["side"] in ("edge", "tab") and expected["side"] != r["side"]
                    and abs(r["_row"].length_mm - expected["_row"].length_mm) <= LENGTH_MM)
            found.append({"piece": r["piece"], "x_mm": r["x_mm"], "y_mm": r["y_mm"], "side": r["side"],
                          "length_mm": r["length_mm"], "read": r["number"], "ocr_conf": r["ocr_conf"],
                          "suggested": number, "partner_piece": expected["piece"], "partner_x_mm": expected["x_mm"],
                          "partner_y_mm": expected["y_mm"], "partner_side": expected["side"],
                          "partner_length_mm": expected["length_mm"],
                          "length_diff_mm": f"{abs(r['_row'].length_mm - expected['_row'].length_mm):.1f}",
                          "why": f"mirror: twin {twin} pairs {source['number']} with {source['partner_piece']}",
                          "sure": "yes" if sure else "no"})
        return found

    def fold_gaps(self) -> list[dict]:
        """Folds of one twin with no fold at their mirror place on the other (issue `missing`: the
        piece missing it, the twin that has it, its kind, its ends on the missing piece in mm), and
        mirror folds read in different styles (issue `style differs`, the twin's style given).
        The twins are laid on each other by the midpoints of their matched numbered lines."""
        gaps = []
        for t in self.twins:
            pairs = [(np.mean(x.edge, axis=0) * self.mm[t.a], np.mean(y.edge, axis=0) * self.mm[t.b]) for x, y in t.lines]
            if len(pairs) < 3:
                continue
            fit = _similar(np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]))
            if fit is None:
                continue
            forward, back = fit
            for source, target, move in ((t.a, t.b, forward), (t.b, t.a, back)):
                corners = self.outline[target]
                for a, b, kind in self.folds[source]:
                    ma, mb = move(a), move(b)
                    # Ends land on the twin's corners or fold ends; a mirrored fold off the piece is no use.
                    ends = np.vstack([corners] + [np.array([p, q]) for p, q, _ in self.folds[target]])
                    ma = min(ends, key=lambda c: np.linalg.norm(c - ma)) if np.min(np.linalg.norm(ends - ma, axis=1)) <= FOLD_MM else ma
                    mb = min(ends, key=lambda c: np.linalg.norm(c - mb)) if np.min(np.linalg.norm(ends - mb, axis=1)) <= FOLD_MM else mb
                    same = [k for p, q, k in self.folds[target]
                            if min(np.linalg.norm(p - ma) + np.linalg.norm(q - mb), np.linalg.norm(p - mb) + np.linalg.norm(q - ma))
                            <= 2 * FOLD_MM]
                    if same:
                        # Mirror folds bend the same way; differing styles mean one was misread.
                        if same[0] != kind and source < target:
                            gaps.append({"piece": target, "twin": source, "kind": kind, "issue": "style differs",
                                         "ax_mm": round(float(ma[0]), 2), "ay_mm": round(float(ma[1]), 2),
                                         "bx_mm": round(float(mb[0]), 2), "by_mm": round(float(mb[1]), 2),
                                         "twin_fold": (a, b)})
                        continue
                    if _to_outline(corners, (ma + mb) / 2) <= ON_CUT_MM or np.linalg.norm(mb - ma) < 2.0:
                        continue
                    # A fold runs inside its piece; one landing outside is the twin's glue tab or a bad fit.
                    polygon = corners.astype(np.float32).reshape(-1, 1, 2)
                    inside = lambda p: cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), True)
                    if inside((ma + mb) / 2) <= ON_CUT_MM or min(inside(ma), inside(mb)) < -ON_CUT_MM:
                        continue
                    gaps.append({"piece": target, "twin": source, "kind": kind, "issue": "missing",
                                 "ax_mm": round(float(ma[0]), 2), "ay_mm": round(float(ma[1]), 2),
                                 "bx_mm": round(float(mb[0]), 2), "by_mm": round(float(mb[1]), 2)})
        return gaps

    def corner_moves(self, clean: set[str] | None = None) -> list[dict]:
        """Twins have the same faces, so a face corner and its mirror corner that sit 0.2-1.5 mm
        apart (after laying the twins on each other) both move to their mirrored mean. When only
        one twin is `clean` (folds in 3D with every glue closed and no corner off), it is taken as
        the truth: the other twin's corners move onto its mirror, up to 3 mm. Rows use the
        piece-edits.csv columns, in page mm. Tab corners find no mirror corner and stay."""
        clean = clean or set()
        rows = []
        for t in self.twins:
            pairs = [(np.mean(x.edge, axis=0) * self.mm[t.a], np.mean(y.edge, axis=0) * self.mm[t.b]) for x, y in t.lines]
            if len(pairs) < 3:
                continue
            fit = _similar(np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]))
            if fit is None:
                continue
            ca, cb = self.outline[t.a], self.outline[t.b]
            truth = ({t.a, t.b} & clean) if len({t.a, t.b} & clean) == 1 else set()
            reach = COPY_MAX_MM if truth else MOVE_MAX_MM
            matched = _mutual(fit[0](ca), cb, MOVE_MAX_MM)
            if len(matched) < 3:
                continue
            # Refit on the matched corners themselves: tighter than the number positions.
            fit = _similar(ca[[i for i, _ in matched]], cb[[j for _, j in matched]])
            if fit is None:
                continue
            forward, back = fit
            for i, j in _mutual(forward(ca), cb, reach):
                gap = float(np.linalg.norm(forward(ca[i]) - cb[j]))
                if gap < MOVE_MIN_MM:
                    continue
                mean = ca[i] if t.a in truth else back(cb[j]) if t.b in truth else (ca[i] + back(cb[j])) / 2
                for piece, twin, old, new in ((t.a, t.b, ca[i], mean), (t.b, t.a, cb[j], forward(mean))):
                    if piece in truth:
                        continue
                    rows.append({"piece": piece, "action": "move", "kind": "",
                                 "ax_mm": round(float(old[0]), 2), "ay_mm": round(float(old[1]), 2),
                                 "bx_mm": round(float(new[0]), 2), "by_mm": round(float(new[1]), 2),
                                 "note": f"mirror of {twin}: corners {gap:.2f} mm apart"
                                         + (f", {twin} folds cleanly in 3D" if truth else "")})
        return rows

    def write(self, path: Path, records: list[dict]) -> int:
        """Every numbered line beside its mirror line; returns how many disagree in length."""
        by_label = {id(r["_row"].label): r for r in records}
        off = 0
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["piece", "twin", "mirror", "number", "side", "length_mm", "twin_number", "twin_side",
                             "twin_length_mm", "length_diff_mm", "check"])
            for t in self.twins:
                for x, y in t.lines:
                    rx, ry = by_label.get(id(x)), by_label.get(id(y))
                    if not rx or not ry:
                        continue
                    diff = abs(rx["_row"].length_mm - ry["_row"].length_mm)
                    off += diff > LENGTH_MM
                    writer.writerow([t.a, t.b, "exact" if t.exact else "variation", rx["number"], rx["side"],
                                     rx["length_mm"], ry["number"], ry["side"],
                                     ry["length_mm"], f"{diff:.1f}", "traced line differs" if diff > LENGTH_MM else "ok"])
        return off


def _similar(a: np.ndarray, b: np.ndarray):
    """Rotation, reflection, scale and shift that best lay points a onto b; returns the maps
    a->b and b->a, or None when the points do not fit within FIT_MM (RMS)."""
    ca, cb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((b - cb).T @ (a - ca))
    rotation = u @ vt
    scale = sv.sum() / max(((a - ca) ** 2).sum(), 1e-9)
    if np.sqrt((((a - ca) @ rotation.T * scale + cb - b) ** 2).sum(1).mean()) > FIT_MM:
        return None
    return (lambda p: (p - ca) @ rotation.T * scale + cb), (lambda q: (q - cb) @ rotation / scale + ca)


def _mutual(a: np.ndarray, b: np.ndarray, reach: float) -> list[tuple[int, int]]:
    """Index pairs of points of a and b that are each other's nearest, within reach (mm)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return [(i, j) for i, j in enumerate(d.argmin(1)) if d[:, j].argmin() == i and d[i, j] <= reach]


def _to_outline(outline: np.ndarray, point: np.ndarray) -> float:
    """Distance from point to the closed polygon's edges."""
    a, b = outline, np.roll(outline, -1, axis=0)
    d = b - a
    t = np.clip(((point - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - point, axis=1)))


def _sequence(a: np.ndarray, b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Line up two runs of line lengths in order, skipping lines on either side.
    Agreeing lengths score 2, lengths within NEAR_MM score 1 (same line, traced off)."""
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(a[i - 1] - b[j - 1])
            gain = 2 if d <= LENGTH_MM else 1 if d <= NEAR_MM else -np.inf
            score[i, j] = max(score[i - 1, j], score[i, j - 1], score[i - 1, j - 1] + gain)
    pairs, i, j = [], n, m
    while i and j:
        if score[i, j] == score[i - 1, j]:
            i -= 1
        elif score[i, j] == score[i, j - 1]:
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return float(score[n, m]), pairs[::-1]


def _along(outline: np.ndarray, point: np.ndarray) -> float:
    """Distance along the outline to the spot nearest `point`."""
    closed = np.vstack([outline, outline[:1]])
    best, walked, at = np.inf, 0.0, 0.0
    for p, q in zip(closed, closed[1:]):
        seg = q - p
        length = float(np.hypot(*seg))
        t = 0.0 if length == 0 else float(np.clip(np.dot(point - p, seg) / length ** 2, 0, 1))
        d = float(np.hypot(*(p + t * seg - point)))
        if d < best:
            best, at = d, walked + t * length
        walked += length
    return at


def _perimeter(outline: np.ndarray) -> float:
    return float(np.hypot(*np.diff(np.vstack([outline, outline[:1]]), axis=0).T).sum())


def _resample(outline: np.ndarray) -> np.ndarray:
    closed = np.vstack([outline, outline[:1]])
    walked = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    t = np.linspace(0, walked[-1], SAMPLES, endpoint=False)
    points = np.c_[np.interp(t, walked, closed[:, 0]), np.interp(t, walked, closed[:, 1])]
    return points - points.mean(axis=0)
