"""Stage 2: find pieces, folds and edge numbers on a rendered pattern page.

The template prints three kinds of ink, which this detector separates:

* solid grey cut lines, long connected strokes that enclose each piece;
* dashed (valley) and dash-dot (mountain) grey fold lines, many short strokes;
* black edge numbers, the darkest pixels on the page.

Piece interiors are the background areas fully enclosed by cut lines. Anything that
is not inside a piece (credits, instructions, an outline with a gap) is kept as
stray artwork so it shows up on the Review layer instead of silently disappearing.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .model import Fold, FoldKind, Label, PageLayout, Piece, Point
from .render import PageImage
from .skeleton import paths_from_skeleton, simplify, thin

DASH_DOT_SHARE = 0.25   # share of long dash runs from which a fold reads as dash-dot


@dataclass
class DetectorSettings:
    ink_threshold: int = 235        # grey level below which a pixel is printed line art
    text_threshold: int = 110       # edge numbers are printed black, cut/fold lines grey
    solid_min_mm: float = 4.3       # strokes longer than this are cut lines, shorter are dashes
    gap_close_mm: float = 0.7       # bridge small breaks in cut lines
    min_piece_mm2: float = 30.0
    fold_min_mm: float = 2.8
    fold_gap_mm: float = 2.0        # longest gap between two dashes of the same fold
    fold_join_mm: float = 8.0       # longest gap joining collinear fold pieces (a number printed on a fold)
    snap_mm: float = 1.5            # fold ends closer than this to a cut vertex/edge snap to it
    fold_min_ink: float = 0.6       # share of a fold line that must be printed dashes
    fold_extend_mm: float = 6.0     # folds stretched further than this to reach the cut are flagged
    outline_tolerance_px: float = 1.5
    notch_mm: float = 6.0           # longest bump (a number touching the cut) straightened away
    notch_depth_mm: float = 2.0     # deepest such bump
    footer_mm: float = 10.0         # right page margin with the credit line and paper note
    label_reach_mm: float = 3.0     # a number further than this from every line marks none
    min_edge_mm: float = 0.8        # shorter cut edges are merged into one corner
    min_corner_deg: float = 4.0     # an outline that turns less than this is still one straight edge
    fitted_corner_mm: float = 2.0
    corner_mm: float = 1.2          # cut edges shorter than this between long edges are rounded corners


def _intersect(p0, p1, p2, p3):
    """Intersection of the infinite lines p0-p1 and p2-p3, or None if parallel."""
    d1, d2 = p1 - p0, p3 - p2
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((p2[0] - p0[0]) * d2[1] - (p2[1] - p0[1]) * d2[0]) / denom
    return p0 + d1 * t


def _angle(u, v) -> float:
    """Unsigned angle between two direction vectors, in radians."""
    cos = float(u @ v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
    return math.acos(max(-1.0, min(1.0, cos)))


def _kernel(size: int) -> np.ndarray:
    return np.ones((max(1, size), max(1, size)), np.uint8)


class PageDetector:
    def __init__(self, settings: DetectorSettings | None = None, folds: dict[str, FoldKind] | None = None):
        self.s = settings or DetectorSettings()
        # Which fold each line style means; set per template in the project YAML.
        self.folds = folds or {"dash_dot": FoldKind.MOUNTAIN, "dashed": FoldKind.VALLEY}

    def _masks(self, page: PageImage):
        """Split the page ink into solid cut lines, number glyphs and fold dashes."""
        px = 1.0 / page.mm_per_px
        rgb = page.rgb.astype(np.int32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.int32)
        # Red instructions ("Print on red paper") are not part of any piece.
        red = (r > g * 1.3) & (r > b * 1.2) & (r > 100)
        ink = ((gray < self.s.ink_threshold) & ~red).astype(np.uint8)
        text_core = (gray < self.s.text_threshold) & ~red

        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        darkest = np.full(count, 255, np.int32)
        np.minimum.at(darkest, labels.ravel(), gray.ravel())
        extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        is_long = extent > self.s.solid_min_mm * px
        is_long[0] = False
        is_glyph = ~is_long & (darkest < self.s.text_threshold)
        is_glyph[0] = False

        solid = is_long[labels].astype(np.uint8)
        glyph = (is_glyph[labels] | text_core).astype(np.uint8)
        dash = (ink.astype(bool) & ~is_long[labels] & ~glyph.astype(bool)).astype(np.uint8)
        return gray, red, ink, text_core, labels, is_long, solid, glyph, dash

    def fold_kind(self, page: PageImage, a: Point, b: Point) -> FoldKind | None:
        """Style of the printed fold between a and b (page px), or None when too little
        dash ink lies along it to tell (a fold drawn where the scan shows none)."""
        if getattr(self, "_dash_page", None) is not page:
            self._dash_page, self._dash = page, self._masks(page)[-1]
        px = 1.0 / page.mm_per_px
        steps = max(2, int(math.dist(a, b) * 2))
        xs = np.linspace(a[0], b[0], steps).round().astype(int).clip(0, self._dash.shape[1] - 1)
        ys = np.linspace(a[1], b[1], steps).round().astype(int).clip(0, self._dash.shape[0] - 1)
        if cv2.dilate(self._dash, _kernel(3))[ys, xs].mean() < 0.3:
            return None
        return self._fold_kind(self._dash, a, b, px)

    def detect(self, page: PageImage, material: str) -> PageLayout:
        px = self.px = 1.0 / page.mm_per_px
        gray, red, ink, text_core, labels, is_long, solid, glyph, dash = self._masks(page)

        # The right margin holds the credit and "Print on ... paper"; it is read, not traced.
        footer_x = gray.shape[1] - round(self.s.footer_mm * px)
        regions = [r for r in self._piece_regions(solid, glyph, px) if cv2.boundingRect(r)[0] < footer_x]
        outlines, holes = self._outlines(regions, solid)
        pieces: list[Piece] = []
        owner = np.zeros(gray.shape, np.int32)  # 1-based piece index per pixel
        for index, (region, outline) in enumerate(zip(regions, outlines)):
            if outline is None:
                continue
            piece = Piece(f"piece-{page.number:02d}-{len(pieces) + 1:02d}", material, outline)
            piece.holes = holes[index]
            inside = cv2.erode(region, _kernel(5))
            piece.folds = self._folds(dash & inside, (dash | glyph) & inside, outline, px)
            piece.outline = self._split_at_folds(outline, piece.folds, px)
            owner[region > 0] = len(pieces) + 1
            pieces.append(piece)

        # Cut ink a traced line already explains (a neighbouring edge, this corner's other edge
        # where it leaves at a shallow angle, or a fold) is not a clipped corner.
        traced = np.zeros_like(solid)
        cv2.polylines(traced, [np.round(np.asarray(p.outline)).astype(np.int32) for p in pieces], True, 1,
                      max(3, round(px)))
        # A fold's first dash often merges with the cut ink at its corner.
        cv2.polylines(traced, [np.round(np.asarray([f.a, f.b])).astype(np.int32) for p in pieces for f in p.folds],
                      False, 1, max(3, round(px)))
        for piece in pieces:
            piece.ink_past = self._ink_past(piece.outline, solid & (1 - traced), px)
        layout = PageLayout(page.number, material, pieces)
        layout.footer = page.rgb[:, footer_x:].copy()
        for label in self._labels(gray, text_core, glyph, px):
            x, y, w, h = label.box
            if x >= footer_x:
                continue
            hit = owner[y + h // 2, x + w // 2] or self._nearest_piece(pieces, label.center, 2 * px)
            (pieces[hit - 1].labels if hit else layout.stray_labels).append(label)
        for piece in pieces:
            for label in piece.labels:
                self._mark_edge(label, piece, px)
        covered = cv2.dilate((owner > 0).astype(np.uint8), _kernel(7))
        stray = ((solid | dash) & (covered == 0)).astype(bool)
        stray[:, footer_x:] = False
        layout.stray_lines = self._centerlines(stray, min_length=int(px))
        return layout

    @staticmethod
    def _split_at_folds(outline: list[Point], folds: list[Fold], px: float, reach_mm: float = 1.0,
                        corner_mm: float = 1.5) -> list[Point]:
        """Put back corners where a fold meets the cut line. When the two cut edges either
        side of such a corner are nearly in line, simplification drops the corner and the
        two edges read as one, too long for the number that marks either of them."""
        points = [tuple(map(float, p)) for p in outline]
        for fold in folds:
            for end in (np.asarray(fold.a, np.float64), np.asarray(fold.b, np.float64)):
                n = len(points)
                best = None
                for i in range(n):
                    a, b = np.asarray(points[i]), np.asarray(points[(i + 1) % n])
                    length = float(np.linalg.norm(b - a))
                    if length < 2 * corner_mm * px:
                        continue
                    u = (b - a) / length
                    along = float((end - a) @ u)
                    across = abs(float((end - a) @ np.array([-u[1], u[0]])))
                    if corner_mm * px < along < length - corner_mm * px and across <= reach_mm * px:
                        if best is None or across < best[0]:
                            best = (across, i, tuple(a + along * u))
                if best:
                    points.insert(best[1] + 1, best[2])
        return points

    @staticmethod
    def _ink_past(outline: list[Point], solid: np.ndarray, px: float, start_mm: float = 0.6,
                  reach_mm: float = 3.0) -> list[tuple[Point, float]]:
        """Corners the printed cut runs on past: from each corner, follow both of its edges
        straight on and measure how far the cut ink continues. A clipped corner leaves the
        true one further out, so both edges ending there read short."""
        ink = cv2.dilate(solid, _kernel(3))
        h, w = ink.shape
        points = np.asarray(outline, np.float64)
        n = len(points)
        found = []
        for i in range(n):
            best = 0.0
            for other, onward in ((points[i - 1], points[(i + 1) % n]), (points[(i + 1) % n], points[i - 1])):
                u = points[i] - other
                length = float(np.linalg.norm(u))
                # Running on along the next edge is just that edge (a fold meeting a straight cut).
                if length < 2 * px or _angle(u, onward - points[i]) < math.radians(25):
                    continue
                u /= length
                run = 0.0
                for t in np.arange(start_mm, reach_mm, 0.1):
                    x, y = np.round(points[i] + u * t * px).astype(int)
                    if not (0 <= x < w and 0 <= y < h) or not ink[y, x]:
                        break
                    run = t
                best = max(best, run)
            if best >= start_mm + 0.3:
                found.append((tuple(points[i]), round(best, 1)))
        return found

    @staticmethod
    def _fold_parts(fold: Fold, piece: Piece, px: float, reach_mm: float = 1.0, corner_mm: float = 1.5):
        """A fold line split where other folds or cut corners touch it. A tab fold that runs
        on in line with another fold reads as one long line; the number marks only its own part."""
        a, b = np.asarray(fold.a, np.float64), np.asarray(fold.b, np.float64)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            return [(fold.a, fold.b, True)]
        u = (b - a) / length
        normal = np.array([-u[1], u[0]])
        touches = [p for f in piece.folds if f is not fold for p in (f.a, f.b)] + list(piece.outline)
        cuts = sorted({round(float((np.asarray(p) - a) @ u), 1) for p in touches
                       if abs(float((np.asarray(p) - a) @ normal)) <= reach_mm * px
                       and corner_mm * px < float((np.asarray(p) - a) @ u) < length - corner_mm * px})
        stops = [0.0, *cuts, length]
        return [(tuple(a + s * u), tuple(a + t * u), True) for s, t in zip(stops, stops[1:])]

    def _mark_edge(self, label: Label, piece: Piece, px: float) -> None:
        """Find the line a number labels: a cut edge (no tab) or a fold it sits on (tab side).
        Numbers run along their line, so lines at an angle to the text are penalised."""
        n = len(piece.outline)
        lines = [(piece.outline[i], piece.outline[(i + 1) % n], False) for i in range(n)]
        lines += [part for fold in piece.folds for part in self._fold_parts(fold, piece, px)]
        centre = np.asarray(label.center)
        reading = np.array([math.cos(math.radians(label.angle)), math.sin(math.radians(label.angle))])
        best, best_score = None, self.s.label_reach_mm * px
        for a, b, tab in lines:
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            length = np.linalg.norm(b - a)
            if length < 1e-6:
                continue
            u = (b - a) / length
            along = float((centre - a) @ u)
            if not 0 <= along <= length:
                continue
            across = abs(float((centre - a) @ np.array([-u[1], u[0]])))
            # A box not clearly longer than wide (a lone digit, or two staggered digits) says nothing
            # about which way the number reads, so its angle is not trusted then.
            turned = (label.glyphs != 1 and getattr(label, "elongation", 2.0) >= 1.8
                      and abs(float(reading @ u)) <= math.cos(math.radians(25)))
            score = across + (2 * px if turned else 0)
            if score < best_score:
                best, best_score = (tuple(a), tuple(b), tab), score
        if best:
            label.edge, label.tab = (best[0], best[1]), best[2]

    @staticmethod
    def _nearest_piece(pieces: list[Piece], point: Point, reach: float) -> int:
        """1-based index of a piece whose cut line passes within `reach` of point, else 0."""
        for index, piece in enumerate(pieces, start=1):
            polygon = np.asarray(piece.outline, np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(polygon, point, True) >= -reach:
                return index
        return 0

    # -- pieces -----------------------------------------------------------------

    def _piece_regions(self, solid: np.ndarray, glyph: np.ndarray, px: float) -> list[np.ndarray]:
        """Flood-fill the paper around the pieces from the page border; whatever the fill
        cannot reach is a piece, cut line included. Folds, numbers and fold dashes that
        touch the cut line from inside then cannot carve notches into a piece."""
        closed = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, _kernel(round(self.s.gap_close_mm * px)))
        height, width = closed.shape
        count, background = cv2.connectedComponents(1 - closed, connectivity=4)
        border = np.unique(np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]]))
        outside = np.isin(background, border[border > 0])
        # A number printed across the cut line belongs to the piece, not the paper around it.
        outside &= ~cv2.dilate(glyph, _kernel(3)).astype(bool)
        # Pieces may touch at a corner, so each enclosed open area seeds its own piece and
        # grows through everything the fill did not reach: cut line, corner wedges, numbers.
        seeds = np.zeros(closed.shape, np.float32)
        count = 0
        for i in np.unique(background):
            if i == 0 or i in border:
                continue
            area = background == i
            if area.sum() >= self.s.min_piece_mm2 * px * px:
                count += 1
                seeds[area] = count
        inside = ~outside
        for _ in range(round(2 * px)):
            grown = np.where(inside & (seeds == 0), cv2.dilate(seeds, _kernel(3)), seeds)
            if np.array_equal(grown, seeds):
                break
            seeds = grown
        return [cv2.morphologyEx((seeds == i).astype(np.uint8), cv2.MORPH_OPEN, _kernel(3))
                for i in range(1, count + 1)]

    def _outlines(self, regions: list[np.ndarray], solid: np.ndarray):
        """Outer cut polygon per region; regions enclosed by another piece become its holes."""
        polygons = []
        on_cut = cv2.dilate(solid, _kernel(5))  # within 2 px of a printed cut line
        for region in regions:
            contours, _ = cv2.findContours(cv2.erode(region, _kernel(3)), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            # Low-poly papercraft: the cut is corners joined by straight lines, nothing curved.
            polygon = cv2.approxPolyDP(contour, self.s.outline_tolerance_px, True)[:, 0, :].astype(np.float64)
            polygon = self._sharpen(self._drop_flat(self._straighten(polygon)))
            # Straight edges follow the printed cut line itself, not the filled region's border,
            # which numbers and fold dashes touching the line can push in or out.
            x, y, w, h = cv2.boundingRect(contour)
            ys, xs = np.nonzero(solid[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4])
            ink = np.column_stack([xs + max(x - 4, 0), ys + max(y - 4, 0)]).astype(np.float64)
            for _ in range(2):
                polygon = self._fit_edges(polygon, ink, band=4.0)
                polygon = self._follow_cut(polygon, on_cut)
            polygon = self._merge_close(polygon, self.s.min_edge_mm * self.px)
            # Fitted edges are reliable, so a clipped acute tip can be restored from further away.
            polygon = self._sharpen(polygon, self.s.fitted_corner_mm)
            # Fitting can leave a vertex where the cut barely turns; corners where a fold meets the cut come back later.
            polygon = self._drop_flat(polygon)
            polygons.append(polygon.astype(np.float32).reshape(-1, 1, 2))
        outlines: list[list[Point] | None] = []
        holes: list[list[list[Point]]] = [[] for _ in regions]
        for i, polygon in enumerate(polygons):
            probe = tuple(float(v) for v in polygon[0][0])
            parent = next((j for j, other in enumerate(polygons)
                           if j != i and cv2.contourArea(other) > cv2.contourArea(polygon)
                           and cv2.pointPolygonTest(other, probe, False) > 0), None)
            points = [(float(x), float(y)) for x, y in polygon[:, 0, :]]
            if parent is None:
                outlines.append(points)
            else:
                holes[parent].append(points)
                outlines.append(None)
        return outlines, holes

    def _straighten(self, polygon: np.ndarray) -> np.ndarray:
        """Remove small bumps where a printed number touches a straight cut line.

        A run of vertices is dropped when the edges on either side of it are in line
        with each other, the run is short, and it strays only a little from the
        straight line that replaces it.
        """
        reach, depth, bend = self.s.notch_mm * self.px, self.s.notch_depth_mm * self.px, math.radians(8)
        changed = True
        while changed and len(polygon) > 4:
            changed = False
            n = len(polygon)
            for i in range(n):
                for span in range(2, 6):  # vertices i+1 .. i+span-1 form the bump
                    j = (i + span) % n
                    if span >= n - 1:
                        break
                    a, b = polygon[i], polygon[j]
                    chord = b - a
                    length = np.linalg.norm(chord)
                    if length == 0 or length > reach:
                        continue
                    before, after = a - polygon[(i - 1) % n], polygon[(j + 1) % n] - b
                    if min(np.linalg.norm(before), np.linalg.norm(after)) < length:
                        continue
                    if _angle(before, chord) > bend or _angle(chord, after) > bend:
                        continue
                    inner = polygon[[(i + k) % n for k in range(1, span)]]
                    normal = np.array([-chord[1], chord[0]]) / length
                    if np.abs((inner - a) @ normal).max() > depth:
                        continue
                    keep = [k for k in range(n) if (k - i) % n == 0 or (k - i) % n >= span]
                    polygon = polygon[keep]
                    changed = True
                    break
                if changed:
                    break
        return polygon

    def _drop_flat(self, polygon: np.ndarray) -> np.ndarray:
        """Remove vertices where the outline barely turns; they are tracing noise, not corners."""
        flat = math.radians(self.s.min_corner_deg)
        changed = True
        while changed and len(polygon) > 3:
            changed = False
            n = len(polygon)
            turns = [_angle(polygon[i] - polygon[i - 1], polygon[(i + 1) % n] - polygon[i]) for i in range(n)]
            i = int(np.argmin(turns))
            if turns[i] < flat:
                polygon = np.delete(polygon, i, axis=0)
                changed = True
        return polygon

    @staticmethod
    def _support(a: np.ndarray, b: np.ndarray, on_cut: np.ndarray) -> float:
        """Share of the segment a-b that runs along printed cut line."""
        steps = max(int(np.linalg.norm(b - a)), 2)
        t = np.linspace(0, 1, steps + 1)[1:-1]
        points = np.round(a + (b - a) * t[:, None]).astype(int)
        h, w = on_cut.shape
        inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        hits = np.zeros(len(points), bool)
        hits[inside] = on_cut[points[inside, 1], points[inside, 0]] > 0
        return float(hits.mean()) if len(hits) else 1.0

    def _follow_cut(self, polygon: np.ndarray, on_cut: np.ndarray, good: float = 0.9) -> np.ndarray:
        """Remove notches: where edges leave the printed cut line, replace up to three
        vertices by a straight chord or by the corner of the edges on either side,
        whichever runs along the cut line."""
        changed = True
        while changed and len(polygon) > 3:
            changed = False
            n = len(polygon)
            support = [self._support(polygon[i], polygon[(i + 1) % n], on_cut) for i in range(n)]
            for i in sorted(range(n), key=lambda i: support[i]):
                if support[i] >= good:
                    break
                best = None
                for k in range(1, min(4, n - 3) + 1):
                    for start in range(i - k + 1, i + 2):  # runs of k vertices touching edge i
                        run = [(start + m) % n for m in range(k)]
                        before, after = polygon[(start - 1) % n], polygon[(start + k) % n]
                        old = min(support[(start - 1 + m) % n] for m in range(k + 1))
                        options = [[]]
                        corner = _intersect(polygon[(start - 2) % n], before, after, polygon[(start + k + 1) % n])
                        if corner is not None and min(np.linalg.norm(corner - polygon[r]) for r in run) < 3 * self.px:
                            options.append([corner])
                        for middle in options:
                            chain = [before, *middle, after]
                            new = min(self._support(p, q, on_cut) for p, q in zip(chain, chain[1:]))
                            if new >= good and new > old + 0.05 and (best is None or new > best[0]):
                                best = (new, run, middle)
                if best:
                    _, run, middle = best
                    keep = [m for m in range(n) if m not in run]
                    head = [polygon[m] for m in keep if m < run[0]]
                    tail = [polygon[m] for m in keep if m > run[0]]
                    if run[-1] < run[0]:  # run wraps past the first vertex
                        head = [polygon[m] for m in keep if run[-1] < m < run[0]]
                        tail = []
                    polygon = np.array(head + list(middle) + tail, np.float64)
                    changed = True
                    break
        return polygon

    @staticmethod
    def _fit_edges(polygon: np.ndarray, contour: np.ndarray, band: float = 2.5) -> np.ndarray:
        """Fit each edge as a straight line through the traced contour near it, then put
        every corner where its two fitted edges meet."""
        n = len(polygon)
        lines = []
        for i in range(n):
            a, b = polygon[i], polygon[(i + 1) % n]
            length = np.linalg.norm(b - a)
            u = (b - a) / max(length, 1e-9)
            offsets = contour - a
            along, across = offsets @ u, offsets @ np.array([-u[1], u[0]])
            near = contour[(np.abs(across) <= band) & (along > band) & (along < length - band)]
            if length < 4 * band or len(near) < 5:
                lines.append((a, u))
                continue
            vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
        corners = polygon.copy()
        for i in range(n):
            (p0, d0), (p1, d1) = lines[i - 1], lines[i]
            corner = _intersect(p0, p0 + d0, p1, p1 + d1)
            # Nearly parallel neighbours give a far-off intersection; keep the traced corner then.
            if corner is not None and np.linalg.norm(corner - polygon[i]) <= 2 * band:
                corners[i] = corner
        return corners

    @staticmethod
    def _merge_close(polygon: np.ndarray, reach: float) -> np.ndarray:
        """Collapse corners closer than `reach` into one: a sliver edge is never a real cut."""
        changed = True
        while changed and len(polygon) > 3:
            changed = False
            n = len(polygon)
            lengths = [np.linalg.norm(polygon[(i + 1) % n] - polygon[i]) for i in range(n)]
            i = int(np.argmin(lengths))
            if lengths[i] < reach:
                polygon[i] = (polygon[i] + polygon[(i + 1) % n]) / 2
                polygon = np.delete(polygon, (i + 1) % n, axis=0)
                changed = True
        return polygon

    def _sharpen(self, polygon: np.ndarray, corner_mm: float | None = None) -> np.ndarray:
        """Restore corners that tracing rounded off: a run of up to three short edges
        between two long ones is replaced by the point where the long edges meet."""
        short = (corner_mm or self.s.corner_mm) * self.px
        changed = True
        while changed and len(polygon) > 3:
            changed = False
            n = len(polygon)
            lengths = [np.linalg.norm(polygon[(i + 1) % n] - polygon[i]) for i in range(n)]
            for i in range(n):  # edge i-1 is long, edges i .. i+k-1 are short, edge i+k is long
                if lengths[i - 1] < short:
                    continue
                k = 0
                while k < 3 and k < n - 3 and lengths[(i + k) % n] < short:
                    k += 1
                after = (i + k) % n
                if k == 0 or lengths[after] < short or lengths[i - 1] < 2 * lengths[i] or lengths[after] < 2 * lengths[after - 1]:
                    continue
                run = [polygon[(i + m) % n] for m in range(k + 1)]
                corner = _intersect(polygon[i - 1], run[0], run[-1], polygon[(after + 1) % n])
                if corner is None or np.linalg.norm(corner - np.mean(run, axis=0)) > 2 * short:
                    continue
                keep = [m for m in range(n) if m not in {(i + j) % n for j in range(1, k + 1)}]
                polygon = polygon.copy()
                polygon[i] = corner
                polygon = polygon[keep]
                changed = True
                break
        return polygon

    # -- folds ------------------------------------------------------------------

    def _folds(self, dashes: np.ndarray, support: np.ndarray, outline: list[Point], px: float) -> list[Fold]:
        found = cv2.HoughLinesP(dashes * 255, 1, np.pi / 720, 12,
                                minLineLength=self.s.fold_min_mm * px,
                                maxLineGap=self.s.fold_gap_mm * px)
        segments = [] if found is None else found.reshape(-1, 4).astype(float).tolist()
        merged = self._merge_collinear(segments, gap=self.s.fold_join_mm * px)
        while True:  # merging can make two longer pieces collinear neighbours
            again = self._merge_collinear([[*a, *b] for a, b in merged], gap=self.s.fold_join_mm * px)
            if len(again) == len(merged):
                break
            merged = again
        polygon = np.asarray(outline, np.float32).reshape(-1, 1, 2)
        # Antialiased cut lines next to numbers can look like dashes; a real fold never runs along the cut.
        merged = [(a, b) for a, b in merged if not self._along_outline(polygon, a, b, 2.5)]
        merged = [self._refit(dashes, a, b) for a, b in merged]
        folds = [Fold(a, b, self._fold_kind(dashes, a, b, px)) for a, b in merged]
        # Numbers printed on a fold interrupt its dashes, so they count as support.
        return self._span_piece(folds, polygon, support, px)

    @staticmethod
    def _merge_collinear(segments, gap: float, tolerance: float = 3.0):
        segments = sorted(segments, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
        merged: list[list[np.ndarray]] = []
        for x1, y1, x2, y2 in segments:
            p, q = np.array([x1, y1]), np.array([x2, y2])
            for line in merged:
                a, b = line
                length = np.linalg.norm(b - a)
                u = (b - a) / length
                normal = np.array([-u[1], u[0]])
                if abs((p - a) @ normal) > tolerance or abs((q - a) @ normal) > tolerance:
                    continue
                lo, hi = sorted(((p - a) @ u, (q - a) @ u))
                if lo < length + gap and hi > -gap:
                    line[0], line[1] = a + u * min(0.0, lo), a + u * max(length, hi)
                    break
            else:
                merged.append([p, q])
        return [(tuple(a), tuple(b)) for a, b in merged]

    @staticmethod
    def _refit(dashes: np.ndarray, a: Point, b: Point, band: float = 3.0) -> tuple[Point, Point]:
        """Least-squares line through the dash pixels near a Hough segment (Hough is quantised)."""
        ys, xs = np.nonzero(dashes)
        points = np.column_stack([xs, ys]).astype(np.float64)
        p, q = np.asarray(a), np.asarray(b)
        length = np.linalg.norm(q - p)
        if length == 0:
            return a, b
        u = (q - p) / length
        offsets = points - p
        along, across = offsets @ u, offsets @ np.array([-u[1], u[0]])
        near = points[(np.abs(across) <= band) & (along >= -band) & (along <= length + band)]
        if len(near) < 10:
            return a, b
        vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d, origin = np.array([vx, vy]), np.array([x0, y0])
        if d @ u < 0:
            d = -d
        ts = (near - origin) @ d
        return tuple(origin + d * ts.min()), tuple(origin + d * ts.max())

    @staticmethod
    def _along_outline(polygon: np.ndarray, a: Point, b: Point, distance: float) -> bool:
        samples = np.linspace(a, b, 9)[1:-1]
        near = [abs(cv2.pointPolygonTest(polygon, (float(x), float(y)), True)) <= distance for x, y in samples]
        return sum(near) > len(near) / 2

    def fold_evidence(self, page: PageImage, a: Point, b: Point) -> float | None:
        """How clearly the printed fold between a and b (page px) reads as dash-dot: the share of
        long dash runs less the dash-dot threshold (above 0: dash-dot, below: dashed), or None
        when too little dash ink lies along it."""
        if self.fold_kind(page, a, b) is None:
            return None
        return self._dash_dot_share(self._dash, a, b, 1.0 / page.mm_per_px) - DASH_DOT_SHARE

    def _dash_dot_share(self, dashes: np.ndarray, a: Point, b: Point, px: float) -> float:
        tolerant = cv2.dilate(dashes, _kernel(3))
        length = math.dist(a, b)
        steps = max(2, int(length * 2))
        xs = np.linspace(a[0], b[0], steps).round().astype(int).clip(0, dashes.shape[1] - 1)
        ys = np.linspace(a[1], b[1], steps).round().astype(int).clip(0, dashes.shape[0] - 1)
        samples = tolerant[ys, xs].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], samples, [0]))))
        runs = (edges[1::2] - edges[::2]) / 2.0          # run lengths in pixels
        interior = runs[1:-1] if len(runs) > 2 else runs  # end runs are often clipped
        # At 180 dpi a dash measures ~11 px; a dash-dot line alternates
        # ~19 px dashes with ~7 px runs where the dot merges with its neighbour.
        if not len(interior):
            return 0.0
        return float((interior >= 15 * px / 7.09).sum()) / len(interior)

    def _fold_kind(self, dashes: np.ndarray, a: Point, b: Point, px: float) -> FoldKind:
        """Dash-dot lines contain short 'dot' runs between the dashes."""
        share = self._dash_dot_share(dashes, a, b, px)
        return self.folds["dash_dot" if share > 0 and share >= DASH_DOT_SHARE else "dashed"]

    def _span_piece(self, folds: list[Fold], polygon: np.ndarray, support: np.ndarray, px: float) -> list[Fold]:
        """Stretch every fold into one straight line from cut edge to cut edge.

        Each fold is extended from its midpoint in both directions to the first cut
        edge it meets, and each end snaps to a cut corner when one is close. A stretched
        line must still be mostly dashes, which rejects Hough lines that cut across a fan
        of folds. Folds that end up on the same pair of edge points are one fold seen twice.
        """
        evidence = cv2.dilate(support, _kernel(5))
        vertices = polygon[:, 0, :].astype(float)
        reach, extend = self.s.snap_mm * px, self.s.fold_extend_mm * px
        spans: list[tuple[Fold, float]] = []
        for fold in folds:
            a, b = np.asarray(fold.a), np.asarray(fold.b)
            mid, length = (a + b) / 2, np.linalg.norm(b - a)
            if length == 0 or cv2.pointPolygonTest(polygon, (float(mid[0]), float(mid[1])), False) <= 0:
                continue
            u = (b - a) / length
            # Crossings inside the detected dashes are corners the fold merely passes near.
            hits = [t for t in self._crossings(mid, u, vertices, reach) if abs(t) >= length / 2 - reach]
            ahead, behind = [t for t in hits if t > 0], [t for t in hits if t < 0]
            if not ahead or not behind:
                continue
            start, end = mid + u * max(behind), mid + u * min(ahead)
            if self._coverage(evidence, start, end) < self.s.fold_min_ink:
                continue
            ends = []
            for point in (start, end):
                distances = np.linalg.norm(vertices - point, axis=1)
                nearest = int(np.argmin(distances))
                ends.append(vertices[nearest] if distances[nearest] <= reach else point)
            stretched = max(abs(max(behind)), min(ahead)) - length / 2
            span = Fold(tuple(ends[0]), tuple(ends[1]), fold.kind, snapped=stretched <= extend)
            if math.dist(span.a, span.b) >= self.s.fold_min_mm * px:
                spans.append((span, length))
        # Keep the best-supported copy of each fold.
        kept: list[tuple[Fold, float]] = []
        for span, support in sorted(spans, key=lambda item: -item[1]):
            same = any(
                (math.dist(span.a, k.a) <= reach and math.dist(span.b, k.b) <= reach)
                or (math.dist(span.a, k.b) <= reach and math.dist(span.b, k.a) <= reach)
                for k, _ in kept)
            if not same:
                kept.append((span, support))
        return [span for span, _ in kept]

    @staticmethod
    def _coverage(mask: np.ndarray, a, b) -> float:
        """Share of the line a-b (ends excluded) that lies on mask pixels."""
        steps = max(2, int(math.dist(a, b)))
        xs = np.linspace(a[0], b[0], steps)[1:-1].round().astype(int).clip(0, mask.shape[1] - 1)
        ys = np.linspace(a[1], b[1], steps)[1:-1].round().astype(int).clip(0, mask.shape[0] - 1)
        return float(mask[ys, xs].mean()) if len(xs) else 0.0

    @staticmethod
    def _crossings(origin: np.ndarray, u: np.ndarray, vertices: np.ndarray, corner_reach: float) -> list[float]:
        """Distances along origin + t*u at which the line crosses the outline."""
        hits = []
        for p, q in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = q - p
            denom = u[0] * edge[1] - u[1] * edge[0]
            if abs(denom) < 1e-9:
                continue
            offset = p - origin
            t = (offset[0] * edge[1] - offset[1] * edge[0]) / denom
            s = (offset[0] * u[1] - offset[1] * u[0]) / denom
            if 0 <= s <= 1:
                hits.append(float(t))
        # A fold that ends on a corner can slip between its two edges; count corners near the line.
        offsets = vertices - origin
        across = np.abs(offsets @ np.array([-u[1], u[0]]))
        hits.extend(float(t) for t in (offsets @ u)[across <= corner_reach])
        return hits

    # -- labels -----------------------------------------------------------------

    def _labels(self, gray: np.ndarray, text_core: np.ndarray, glyph: np.ndarray, px: float):
        """Group glyphs into numbers; keep each one's crop, centre, direction and size.
        Two numbers printed against each other merge into one group about two digits high;
        their digits stand in two rows, so such a group is split into one number per row."""
        grouped = cv2.dilate(glyph, _kernel(round(px)))
        count, groups, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        found = []
        for i in range(1, count):
            x, y, w, h, _ = stats[i]
            mask = groups[y:y + h, x:x + w] == i
            core = mask & text_core[y:y + h, x:x + w]
            ys, xs = np.nonzero(core)
            if len(xs) < 5:
                continue
            _, size, _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            found.append((x, y, mask, core, min(size)))
        height = float(np.median([f[4] for f in found])) if found else 0.0
        for x, y, mask, core, across in found:
            parts, part_of, sizes, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
            glyphs = [j for j in range(1, parts) if sizes[j, cv2.CC_STAT_AREA] >= 5]
            rows = self._rows(part_of, glyphs, sizes) if across > 1.6 * height else [glyphs]
            for row in rows:
                own = mask
                if len(rows) > 1:
                    own = cv2.dilate(np.isin(part_of, row).astype(np.uint8), _kernel(3)).astype(bool) & mask
                label = self._label(gray, text_core, own, x, y, len(row), crop=len(rows) > 1)
                if label is not None:
                    yield label

    @staticmethod
    def _label(gray: np.ndarray, text_core: np.ndarray, mask: np.ndarray, x: int, y: int, glyphs: int,
               crop: bool = False) -> Label | None:
        ys, xs = np.nonzero(mask & text_core[y:y + mask.shape[0], x:x + mask.shape[1]])
        if len(xs) < 5:
            return None
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        if rw < rh:  # make the long side the reading direction
            angle, rw, rh = angle + 90, rh, rw
        if not crop:
            ink = np.where(mask, 255 - gray[y:y + mask.shape[0], x:x + mask.shape[1]], 0).clip(0, 255).astype(np.uint8)
            return Label((x, y, mask.shape[1], mask.shape[0]), (float(cx + x), float(cy + y)), float(angle), float(rh), ink,
                         glyphs=glyphs, elongation=float(rw / max(rh, 1e-6)))
        # Crop to this number's own pixels, so a split-off neighbour is not read with it.
        top, left = int(ys.min()), int(xs.min())
        h, w = int(ys.max()) - top + 1, int(xs.max()) - left + 1
        pad = 2
        top, left = max(top - pad, 0), max(left - pad, 0)
        h, w = min(h + 2 * pad, mask.shape[0] - top), min(w + 2 * pad, mask.shape[1] - left)
        window = mask[top:top + h, left:left + w]
        ink = np.where(window, 255 - gray[y + top:y + top + h, x + left:x + left + w], 0).clip(0, 255).astype(np.uint8)
        return Label((x + left, y + top, w, h), (float(cx + x), float(cy + y)), float(angle), float(rh), ink, glyphs=glyphs,
                     elongation=float(rw / max(rh, 1e-6)))

    @staticmethod
    def _rows(part_of: np.ndarray, glyphs: list[int], sizes: np.ndarray, min_area: int = 25) -> list[list[int]]:
        """Glyphs in rows of text. A digit is taller than wide, so the digits' long axes give the
        direction across the text; glyphs whose extents along it overlap stand in one row.
        Specks too small to be a digit join the nearest row."""
        if len(glyphs) < 2:
            return [glyphs]
        pixels = {j: np.column_stack(np.nonzero(part_of == j)[::-1]).astype(np.float32) for j in glyphs}
        doubled, heights = [], []
        for j, p in pixels.items():
            if len(p) < 20:
                continue
            _, (w, h), angle = cv2.minAreaRect(p)
            if max(w, h) >= 1.3 * max(min(w, h), 1):
                tall = angle + (90 if w < h else 0)  # direction of the long side, degrees
                doubled.append(np.exp(2j * np.radians(tall)))
                heights.append(max(w, h))
        if not doubled:
            return [glyphs]
        across = np.angle(np.mean(doubled)) / 2
        normal = np.array([np.cos(across), np.sin(across)])
        spans = {j: (float((p @ normal).min()), float((p @ normal).max())) for j, p in pixels.items()}
        parent = {j: j for j in glyphs}
        def root(j):
            while parent[j] != j:
                j = parent[j]
            return j
        for m, a in enumerate(glyphs):
            for b in glyphs[m + 1:]:
                (a0, a1), (b0, b1) = spans[a], spans[b]
                if min(a1, b1) - max(a0, b0) >= 0.5 * min(a1 - a0, b1 - b0):
                    parent[root(a)] = root(b)
        grouped = collections.defaultdict(list)
        for j in glyphs:
            grouped[root(j)].append(j)
        rows = list(grouped.values())
        big = [r for r in rows if sum(sizes[j, cv2.CC_STAT_AREA] for j in r) >= min_area]
        if len(big) < 2:
            return [glyphs]
        centre = lambda r: np.mean([sum(spans[j]) / 2 for j in r])
        # Rows of one number sit on one line; two stacked numbers are a whole digit height apart.
        lines = sorted(centre(r) for r in big)
        if min(b - a for a, b in zip(lines, lines[1:])) < 0.8 * float(np.median(heights)):
            return [glyphs]
        for r in rows:
            if r not in big:
                min(big, key=lambda k: abs(centre(k) - centre(r))).extend(r)
        return big

    @staticmethod
    def _centerlines(mask: np.ndarray, min_length: int, offset: tuple[int, int] = (0, 0)) -> list[list[Point]]:
        if not mask.any():
            return []
        ox, oy = offset
        lines = []
        for chain, _ in paths_from_skeleton(thin(mask), min_length=min_length):
            points = simplify(chain, tolerance=0.6)
            if len(points) >= 2:
                lines.append([(float(x + ox), float(y + oy)) for y, x in points])
        return lines
