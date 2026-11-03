"""Fit the folded mask to a wearer's head and print where to cut the sight holes (issue #79).

Reads the folded model (data/<m>/fold3d.json, millimetres) and the YAML `fit:` block:

    fit:
      head: female            # child | female | male preset
      circumference_mm: 545   # optional, scales the preset
      eye_distance_mm: 61     # optional, pupil to pupil
      clearance_mm: 10        # air between head and card
      sight_hole_mm: 30       # diameter of each sight hole
      eyes: openings          # pupils centred behind the model's eye openings (or: free)
      up: +y                  # roughly up in the model's axes
      front: eyes             # the way the eye openings face (or an axis such as -y)

The head is two ellipsoids (cranium, face and jaw) and a nose, placed relative to the pupils. With
`eyes: openings` (the default, when the model has a mirrored pair of openings in front) the mask is
scaled so its eye openings sit centred over the pupils, and the head room is reported as a check.
With `eyes: free` the mask is scaled until the head, with clearance, just fits inside the card. The
parts of the head that leave through the neck opening (the largest one) are allowed either way. Then a ray from each pupil, straight ahead, finds where
the wearer looks out: through an opening, or through a piece where a sight hole must be cut.

Writes data/<m>/head-fit.json, head-fit-views.svg (front and side, scaled to an A4 page) and
head-fit-eyes.svg (A4 at 1:1: each piece the eyes look through, at the fitted scale, with the
sight hole marked).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

# Rounded adult means in the style of ANSUR II (2012) and a ~8 year old child; mm. They are a
# starting point: a measured head circumference and pupil distance always win.
#   breadth: ear to ear above the ears, length: brow to back of head, height: chin to crown.
HEAD_PRESETS = {
#   circumference: round the head just above the brow and ears, as a tape measure gives it.
    "male":   dict(breadth=154.0, length=199.0, height=236.0, circumference=572.0, eye_distance=64.0),
    "female": dict(breadth=146.0, length=187.0, height=221.0, circumference=549.0, eye_distance=62.0),
    "child":  dict(breadth=140.0, length=177.0, height=200.0, circumference=525.0, eye_distance=56.0),
}

SIGHT_HOLE_MM = 30.0            # default sight hole diameter
A4_MM = (210.0, 297.0)


def _axis(text: str) -> np.ndarray:
    text = str(text).strip().lower()
    sign = -1.0 if text.startswith("-") else 1.0
    vector = np.zeros(3)
    vector["xyz".index(text[-1])] = sign
    return vector


@dataclass(frozen=True)
class HeadSize:
    breadth: float
    length: float
    height: float
    circumference: float
    eye_distance: float
    name: str = "custom"

    @classmethod
    def from_fit(cls, fit: dict) -> "HeadSize":
        name = str(fit.get("head", "female")).lower()
        if name not in HEAD_PRESETS:
            raise ValueError(f"fit.head must be one of {sorted(HEAD_PRESETS)}, not {name!r}")
        head = cls(name=name, **HEAD_PRESETS[name])
        if "circumference_mm" in fit:
            ratio = float(fit["circumference_mm"]) / head.circumference
            head = replace(head, breadth=head.breadth * ratio, length=head.length * ratio,
                           height=head.height * ratio, circumference=head.circumference * ratio,
                           name=f"{name}, measured")
        if "eye_distance_mm" in fit:
            head = replace(head, eye_distance=float(fit["eye_distance_mm"]), name=f"{name}, measured")
        return head


@dataclass(frozen=True)
class Ellipsoid:
    centre: np.ndarray      # head frame: x right, y back, z up, origin between the pupils
    radii: np.ndarray

    def surface(self, count: int) -> np.ndarray:
        i = np.arange(count) + 0.5
        z = 1 - 2 * i / count
        r = np.sqrt(1 - z * z)
        phi = i * math.pi * (3 - math.sqrt(5))
        unit = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return self.centre + unit * self.radii

    def inside(self, points: np.ndarray) -> np.ndarray:
        return (((points - self.centre) / self.radii) ** 2).sum(axis=1) < 1.0


class HeadModel:
    """A head built round the pupils: cranium, face and jaw, nose."""

    def __init__(self, size: HeadSize):
        self.size = size
        b, l, h = size.breadth, size.length, size.height
        brow = -12.0                                     # forehead is this far in front of the pupils
        self.cranium = Ellipsoid(np.array([0.0, brow + l / 2, 0.51 * h - 0.42 * h]),
                                 np.array([b / 2, l / 2, 0.42 * h]))
        self.jaw = Ellipsoid(np.array([0.0, -5.0 + 0.32 * l, -0.49 * h + 0.30 * h]),
                             np.array([0.42 * b, 0.32 * l, 0.30 * h]))
        self.nose = np.array([[0.0, -30.0, -40.0], [0.0, -18.0, -20.0], [0.0, -10.0, -5.0]])
        self.pupils = np.array([[-size.eye_distance / 2, 0.0, 0.0], [size.eye_distance / 2, 0.0, 0.0]])

    def samples(self, count: int = 500) -> np.ndarray:
        """Points on the outside of the head, head frame. The underside of the jaw behind the chin is
        left out: the neck is there, and it runs down and back out of the mask."""
        cranium, jaw = self.cranium.surface(count), self.jaw.surface(count)
        c, r = self.jaw.centre, self.jaw.radii
        throat = (jaw[:, 2] < c[2]) & (jaw[:, 1] > c[1] - 0.3 * r[1])
        return np.vstack([cranium[~self.jaw.inside(cranium)], jaw[~self.cranium.inside(jaw) & ~throat], self.nose])

    def outline(self, view: str, steps: int = 96) -> list[np.ndarray]:
        """Ellipse outlines for a drawing: view 'front' gives (x, z), 'side' gives (y, z)."""
        t = np.linspace(0, 2 * math.pi, steps)
        k = 0 if view == "front" else 1
        return [np.column_stack([e.centre[k] + e.radii[k] * np.cos(t), e.centre[2] + e.radii[2] * np.sin(t)])
                for e in (self.cranium, self.jaw)]


class FoldedMask:
    """The glued model as triangles, turned so x is right, y is back and z is up (head frame axes)."""

    def __init__(self, data: dict, up: np.ndarray, front: np.ndarray | None = None):
        """`front` None: the wearer looks out the way the eye openings face; `up` is then only a hint
        (the part of it square to `front`). The model's own x axis must be its mirror axis."""
        raw = np.asarray(data["vertices"], dtype=float)
        self.faces = [f for f in data["faces"] if not f["tab"]]
        fans = [(k, 0, i, i + 1) for k, f in enumerate(self.faces) for i in range(1, len(f["v"]) - 1)]
        self.tri_face = np.array([k for k, *_ in fans])
        self.tri_corner = np.array([c for _, *c in fans])            # corners of the face, fan-split
        self.triangles = np.array([raw[np.array(self.faces[k]["v"])[c]] for k, *c in fans])
        self.pieces = data["pieces"]
        self.colours = data.get("colours", {})
        self.openings = self._openings(raw)
        if front is None:
            front = self._eye_direction()
            if front is None:
                raise ValueError("no pair of eye openings found: give fit.front (e.g. -y)")
        front = front / np.linalg.norm(front)
        up = up - (up @ front) * front
        up /= np.linalg.norm(up)
        right = np.cross(-front, up)        # looking along `front`, x points to the right hand
        self.frame = np.vstack([right, -front, up])
        self.triangles = self.triangles @ self.frame.T
        self.openings = [o @ self.frame.T for o in self.openings]
        # The neck is the largest opening: the head may pass out through its plane.
        neck = self.openings[0]
        self.neck_centre = neck.mean(0)
        normal = np.linalg.svd(neck - self.neck_centre)[2][2]
        if normal @ (self.neck_centre - self.triangles.reshape(-1, 3).mean(0)) < 0:
            normal = -normal
        self.neck_normal = normal                    # points out of the mask

    def eye_openings(self, tolerance: float = 6.0, front_half: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        """The largest pair of mirrored openings (in the front half): (left, right) corner points."""
        middle = self.triangles.reshape(-1, 3).mean(0)
        loops = self.openings[1:]
        for i, a in enumerate(loops):
            ca = a.mean(0)
            if (front_half and ca[1] > middle[1]) or abs(ca[0]) < 10:
                continue
            for b in loops[i + 1:]:
                cb = b.mean(0)
                if np.linalg.norm(cb - ca * np.array([-1, 1, 1])) < tolerance:
                    return (a, b) if ca[0] < cb[0] else (b, a)
        return None

    def _eye_direction(self) -> np.ndarray | None:
        """Mean outward normal of the eye openings, in model axes, made symmetric."""
        holes = self.eye_openings(front_half=False)
        if holes is None:
            return None
        middle = self.triangles.reshape(-1, 3).mean(0)
        normals = []
        for hole in holes:
            c = hole.mean(0)
            n = np.linalg.svd(hole - c)[2][2]
            normals.append(n if n @ (c - middle) > 0 else -n)
        direction = np.mean(normals, axis=0)
        direction[0] = 0.0
        return direction

    def beyond_neck(self, points: np.ndarray) -> np.ndarray:
        return (points - self.neck_centre) @ self.neck_normal > 0

    def _openings(self, vertices: np.ndarray) -> list[np.ndarray]:
        """Corner points of each open loop of card edges, largest first."""
        edges: dict[tuple[int, int], int] = {}
        weld = self._weld(vertices)
        for face in self.faces:
            v = [weld[i] for i in face["v"]]
            for a, b in zip(v, v[1:] + v[:1]):
                if a != b:
                    key = (min(a, b), max(a, b))
                    edges[key] = edges.get(key, 0) + 1
        loose = [e for e, n in edges.items() if n == 1]
        if not loose:
            raise ValueError("the folded model has no openings: where does the head go in?")
        parent = {v: v for e in loose for v in e}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for a, b in loose:
            parent[find(a)] = find(b)
        loops: dict[int, list] = {}
        for a, b in loose:
            loops.setdefault(find(a), []).append((a, b))
        ordered = sorted(loops.values(), reverse=True,
                         key=lambda es: sum(np.linalg.norm(vertices[a] - vertices[b]) for a, b in es))
        return [vertices[sorted({v for e in es for v in e})] for es in ordered]

    @staticmethod
    def _weld(vertices: np.ndarray, tolerance: float = 1.5) -> list[int]:
        from scipy.spatial import cKDTree
        parent = list(range(len(vertices)))

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for a, b in cKDTree(vertices).query_pairs(tolerance):
            parent[find(a)] = find(b)
        return [find(v) for v in range(len(vertices))]

    def winding(self, points: np.ndarray, chunk: int = 4000) -> np.ndarray:
        """Generalised winding number: about 1 inside, 0 outside, robust to the open eyes and neck."""
        out = np.empty(len(points))
        for start in range(0, len(points), chunk):
            p = points[start:start + chunk, None, None, :]
            a, b, c = (self.triangles[None, :, i, :] - p[:, :, 0, :] for i in range(3))
            la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
            det = np.einsum("pti,pti->pt", a, np.cross(b, c))
            den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
            out[start:start + chunk] = np.arctan2(det, den).sum(axis=1) / (2 * math.pi)
        return np.abs(out)

    def surface_points(self, spacing: float = 2.0) -> np.ndarray:
        points = []
        for tri in self.triangles:
            n = max(2, int(max(np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[0])) / spacing))
            u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
            keep = u + v <= 1
            u, v = u[keep], v[keep]
            points.append(tri[0] + u[:, None] * (tri[1] - tri[0]) + v[:, None] * (tri[2] - tri[0]))
        return np.vstack(points)

    def ray(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, int, np.ndarray] | None:
        """First card hit along a ray: distance, face index and barycentric (u, v)."""
        e1 = self.triangles[:, 1] - self.triangles[:, 0]
        e2 = self.triangles[:, 2] - self.triangles[:, 0]
        h = np.cross(direction, e2)
        det = (e1 * h).sum(1)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        s = origin - self.triangles[:, 0]
        u = (s * h).sum(1) * inv
        q = np.cross(s, e1)
        v = (q @ direction) * inv
        t = (q * e2).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        if not hit.any():
            return None
        best = int(np.argmin(np.where(hit, t, np.inf)))
        return float(t[best]), best, np.array([u[best], v[best]])


class ClearanceField:
    """Signed distance to the card on a grid: positive inside the mask, negative outside (model mm)."""

    def __init__(self, mask: FoldedMask, step: float = 6.0):
        from scipy.interpolate import RegularGridInterpolator
        from scipy.spatial import cKDTree
        low, high = mask.triangles.reshape(-1, 3).min(0) - 2 * step, mask.triangles.reshape(-1, 3).max(0) + 2 * step
        axes = [np.arange(low[k], high[k] + step, step) for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        distance, _ = cKDTree(mask.surface_points()).query(grid)
        # The winding number is slow (every point against every triangle): work it out on a grid twice
        # as coarse and interpolate; it only gives the sign, and near the card the distance is small anyway.
        coarse = [a[::2] for a in axes]
        cgrid = np.stack(np.meshgrid(*coarse, indexing="ij"), axis=-1).reshape(-1, 3)
        winding = RegularGridInterpolator(coarse, mask.winding(cgrid).reshape([len(a) for a in coarse]),
                                          bounds_error=False, fill_value=0.0)
        inside = winding(grid) > 0.5
        values = np.where(inside, distance, -distance).reshape([len(a) for a in axes])
        self._interp = RegularGridInterpolator(axes, values, bounds_error=False, fill_value=-1e3)
        self.low, self.high = low, high

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self._interp(points)


@dataclass
class FitResult:
    head: dict
    scale: float
    clearance_mm: float
    pupils_mm: list            # pupil midpoint in the scaled mask, head frame axes
    tightest_mm: float
    mask_size_mm: list
    eyes: str = "free"                 # "openings": pupils behind the model's eye openings
    eye_scale_range: list | None = None
    sight: list = field(default_factory=list)


class HeadFitter:
    """Smallest mask scale at which the head, with clearance, fits inside the card."""

    def __init__(self, mask: FoldedMask, head: HeadModel, clearance: float, field: "ClearanceField | None" = None):
        self.mask, self.head, self.clearance = mask, head, clearance
        self.field = field or ClearanceField(mask)
        self.samples = head.samples()

    def margin(self, scale: float, pupils: np.ndarray) -> float:
        """Least air between head and card in real mm, ignoring head that has left through the neck."""
        points = (self.samples + pupils) / scale
        keep = ~self.mask.beyond_neck(points)
        return float((scale * self.field(points[keep])).min()) if keep.any() else -1e3

    def best_position(self, scale: float, step: float = 4.0) -> tuple[np.ndarray, float]:
        low, high = self.field.low * scale, self.field.high * scale
        ys, zs = np.arange(low[1], high[1], step), np.arange(low[2], high[2], step)
        grid = np.array([[0.0, y, z] for y in ys for z in zs])
        # the cranium centre must be at least its smallest radius plus clearance from the card
        centre = grid + self.head.cranium.centre
        need = (self.head.cranium.radii.min() + self.clearance) * 0.9
        grid = grid[scale * self.field(centre / scale) >= need]
        if not len(grid):
            return np.zeros(3), -math.inf
        margins = np.array([self.margin(scale, p) for p in grid])
        best, best_margin = grid[int(np.argmax(margins))], float(margins.max())
        for fine in (step / 2, step / 4):         # polish round the best grid point
            for dy in (-fine, 0, fine):
                for dz in (-fine, 0, fine):
                    p = best + np.array([0.0, dy, dz])
                    m = self.margin(scale, p)
                    if m > best_margin:
                        best, best_margin = p, m
        return best, best_margin

    def fit_to_eyes(self, holes: tuple[np.ndarray, np.ndarray],
                    edge_mm: float = 5.0) -> tuple[float, np.ndarray, float, tuple]:
        """Scale so the eye openings sit centred over the pupils (scale = eye distance / opening
        spacing). Returns scale, pupil midpoint (behind the openings, at the best depth), the head's
        least clearance there (a check only: the folded model's inside is rough) and the scale range
        that keeps each pupil at least `edge_mm` inside its opening."""
        centres = np.array([h.mean(0) for h in holes])
        half_gap = (centres[1, 0] - centres[0, 0]) / 2
        half_width = np.mean([np.ptp(h[:, 0]) for h in holes]) / 2
        ipd = self.head.size.eye_distance / 2
        lowest = (ipd + edge_mm) / (half_gap + half_width)
        highest = (ipd - edge_mm) / (half_gap - half_width) if half_gap > half_width else math.inf
        scale = ipd / half_gap
        mid = np.array([0.0, centres[:, 1].mean(), centres[:, 2].mean()])
        best = None
        for relief in np.arange(8.0, 40.0, 3.0):            # pupils this far behind the openings
            p = mid * scale + np.array([0.0, relief, 0.0])
            m = self.margin(scale, p)
            if best is None or m > best[1]:
                best = (p, m)
        return scale, best[0], best[1], (lowest, highest)

    def fit(self, low: float = 0.4, high: float = 2.5, tolerance: float = 0.01) -> tuple[float, np.ndarray, float]:
        pos, m = self.best_position(high)
        if m < self.clearance:
            raise RuntimeError(f"the head does not fit even at {high:.0%}")
        while high - low > tolerance:
            mid = (low + high) / 2
            p, m = self.best_position(mid)
            if m >= self.clearance:
                high, pos = mid, p
            else:
                low = mid
        pos, m = self.best_position(high, step=2.0)
        return high, pos, m


class SightFinder:
    """Where each pupil looks out of the scaled mask."""

    def __init__(self, mask: FoldedMask, scale: float, hole_mm: float = SIGHT_HOLE_MM):
        self.mask, self.scale, self.hole_mm = mask, scale, hole_mm

    def look(self, pupil: np.ndarray) -> dict:
        hit = self.mask.ray(pupil / self.scale, np.array([0.0, -1.0, 0.0]))
        if hit is None:
            return dict(through="opening", pupil_mm=pupil.round(1).tolist())
        t, index, (u, v) = hit
        face = self.mask.faces[self.mask.tri_face[index]]
        flat = np.asarray(face["flat"], dtype=float)[self.mask.tri_corner[index]]
        at = flat[0] + u * (flat[1] - flat[0]) + v * (flat[2] - flat[0])
        distance = t * self.scale
        return dict(through="card", piece=face["piece"], face=face["id"],
                    distance_mm=round(distance, 1),
                    flat_mm=at.round(2).tolist(),          # on the piece as printed (pattern mm, before scaling)
                    hole_radius_mm=round(self.hole_mm / 2, 1),
                    pupil_mm=pupil.round(1).tolist())


class TemplateWriter:
    """The two A4 SVG pages."""

    def __init__(self, mask: FoldedMask, data: dict, head: HeadModel, result: FitResult):
        self.mask, self.data, self.head, self.result = mask, data, head, result

    def _svg(self, body: list[str]) -> str:
        w, h = A4_MM
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}" '
                f'font-family="sans-serif">\n<rect width="{w}" height="{h}" fill="white"/>\n'
                + "\n".join(body) + "\n</svg>\n")

    def views(self) -> str:
        r, s = self.result, self.result.scale
        tri = self.mask.triangles * s
        pupils = np.array(r.pupils_mm)
        head_offset = pupils
        panels = [("front", 0, "front: as seen by someone facing the wearer"), ("side", 1, "side: the wearer looks left")]
        body = [f'<text x="12" y="14" font-size="5">Head fit: {r.head["name"]} head, mask at {s:.0%} '
                f'(head room check: {r.tightest_mm:.0f} mm, wanted {r.clearance_mm:g})</text>',
                f'<text x="12" y="20" font-size="3.2">Mask {r.mask_size_mm[0]:.0f} × {r.mask_size_mm[1]:.0f} × '
                f'{r.mask_size_mm[2]:.0f} mm (wide × deep × tall). Grey: card. Blue: head. Red: pupils and sight.</text>']
        extent = tri.reshape(-1, 3)
        span = max(np.ptp(extent[:, 0]), np.ptp(extent[:, 1]), np.ptp(extent[:, 2]))
        zoom = min(180.0 / span, 125.0 / np.ptp(extent[:, 2]))
        for row, (name, k, caption) in enumerate(panels):
            cx, cy = 105.0, 95.0 + row * 135.0
            mid = (extent.min(0) + extent.max(0)) / 2
            sign = -1.0 if name == "front" else 1.0

            def xy(p, k=k, sign=sign, cx=cx, cy=cy, mid=mid):
                return cx + sign * (p[..., k] - mid[k]) * zoom, cy - (p[..., 2] - mid[2]) * zoom
            depth = 1 if name == "front" else 0
            order = np.argsort(tri[:, :, depth].mean(1) * (1 if name == "front" else -1))
            body.append(f'<g stroke="#999" stroke-width="0.1" fill="#e6e6e6" fill-opacity="0.35">')
            for i in order:
                x, y = xy(tri[i])
                body.append('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + 'Z"/>')
            body.append("</g>")
            for ring in self.head.outline(name):
                p = np.zeros((len(ring), 3))
                p[:, k], p[:, 2] = ring[:, 0], ring[:, 1]
                x, y = xy(p + head_offset)
                body.append('<path fill="none" stroke="#1f5fa8" stroke-width="0.5" d="M'
                            + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + '"/>')
            for pupil, sight in zip(self.head.pupils + head_offset, r.sight):
                x, y = xy(pupil)
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.2" fill="#c62828"/>')
                if name == "side":
                    reach = sight.get("distance_mm", 60.0)
                    x2, y2 = xy(pupil + np.array([0.0, -reach, 0.0]))
                    body.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                                f'stroke="#c62828" stroke-width="0.4" stroke-dasharray="1.5,1"/>')
            body.append(f'<text x="12" y="{cy - 62:.0f}" font-size="3.5">{caption}</text>')
        bar = 50 * zoom
        body.append(f'<line x1="150" y1="285" x2="{150 + bar:.1f}" y2="285" stroke="black" stroke-width="0.5"/>'
                    f'<text x="150" y="282" font-size="3">50 mm</text>')
        return self._svg(body)

    def eye_openings(self) -> str:
        """1:1 face template: both eye openings as the wearer's face meets them, with the pupils."""
        r, s = self.result, self.result.scale
        holes = self.mask.eye_openings()
        pupils = self.head.pupils + np.array(r.pupils_mm)
        cx, cy = 105.0, 120.0
        mid = np.mean([h.mean(0) for h in holes], axis=0) * s

        def xy(p):          # seen from the front, facing the wearer: their right on the page's left
            return cx - (p[..., 0] - mid[0]), cy - (p[..., 2] - mid[2])
        body = [f'<text x="12" y="14" font-size="5">Eye openings at 1:1, mask at {s:.0%}</text>',
                f'<text x="12" y="20" font-size="3.2">{r.head["name"]} head, pupils {r.head["eye_distance"]:.0f} mm apart. '
                'Print at 100 % (no fit to page); the 50 mm bar checks it.</text>',
                '<text x="12" y="25" font-size="3.2">Hold it to the face, red dots on the pupils: the grey '
                'outlines are where the mask\'s eye openings will be.</text>',
                '<line x1="12" y1="31" x2="62" y2="31" stroke="black" stroke-width="0.5"/>'
                '<text x="64" y="32" font-size="3">50 mm</text>']
        for hole in holes:
            ring = hole * s
            angle = np.arctan2(ring[:, 2] - ring[:, 2].mean(), ring[:, 0] - ring[:, 0].mean())
            x, y = xy(ring[np.argsort(angle)])
            body.append('<path fill="#eeeeee" stroke="#555" stroke-width="0.4" d="M'
                        + " L".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y)) + 'Z"/>')
        for pupil in pupils:
            x, y = xy(pupil)
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.5" fill="#c62828"/>')
        (x1, y1), (x2, y2) = (xy(p) for p in pupils)
        body.append(f'<line x1="{x1:.2f}" y1="{y1 + 25:.2f}" x2="{x2:.2f}" y2="{y2 + 25:.2f}" stroke="#c62828" '
                    f'stroke-width="0.3"/><text x="{(x1 + x2) / 2 - 8:.1f}" y="{y1 + 30:.1f}" font-size="3">'
                    f'{r.head["eye_distance"]:.0f} mm</text>')
        lo, hi = r.eye_scale_range or (None, None)
        if lo:
            body.append(f'<text x="12" y="200" font-size="3.5">The pupils stay 5 mm inside the openings from '
                        f'{lo:.0%} to {hi:.0%}; {s:.0%} centres them.</text>')
        body.append(f'<text x="12" y="207" font-size="3.5">Mask at {s:.0%}: {r.mask_size_mm[0]:.0f} mm wide '
                    f'with the horns.</text>')
        return self._svg(body)

    def eyes(self) -> str:
        if self.result.eyes == "openings":
            return self.eye_openings()
        r, s = self.result, self.result.scale
        body = [f'<text x="12" y="14" font-size="5">Sight holes at 1:1, mask at {s:.0%}</text>',
                '<text x="12" y="20" font-size="3.2">Print at 100 % (no fit to page). Lay over the cut piece; '
                'cut out the red circle. The 50 mm bar checks the print size.</text>',
                '<line x1="12" y1="26" x2="62" y2="26" stroke="black" stroke-width="0.5"/>'
                '<text x="64" y="27" font-size="3">50 mm</text>']
        slots = [(12.0, 32.0), (12.0, 165.0)]
        cards = [x for x in r.sight if x["through"] == "card"]
        if not cards:
            body.append('<text x="12" y="45" font-size="4">Both eyes look out through openings: no holes to cut.</text>')
        seen: dict[str, list] = {}
        for sight in cards:
            seen.setdefault(sight["piece"], []).append(sight)
        for (x0, y0), (piece, sights) in zip(slots, seen.items()):
            tris = [np.asarray(f["flat"], float) * s for f in self.data["faces"] if f["piece"] == piece]
            pts = np.vstack(tris + [np.array([st["flat_mm"]]) * s for st in sights])
            lo, hi = pts.min(0), pts.max(0)
            room = np.array([186.0, 120.0])
            if np.any(hi - lo > room):
                lo = np.array(sights[0]["flat_mm"]) * s - room / 2      # piece larger than the slot: centre on the hole
            off = np.array([x0, y0 + 8]) - lo
            colour = self.mask.colours.get(self.mask.pieces.get(piece, {}).get("material", ""), "#888")
            body.append(f'<text x="{x0}" y="{y0 + 4}" font-size="3.5">{piece} ({self.mask.pieces.get(piece, {}).get("material", "?")}), '
                        f'flat side up as printed</text>')
            body.append(f'<g stroke="{colour}" stroke-width="0.25" fill="none">')
            for t in tris:
                q = t + off
                body.append('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in q) + 'Z"/>')
            body.append("</g>")
            for st in sights:
                c = np.array(st["flat_mm"]) * s + off
                body.append(f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="{st["hole_radius_mm"]:.2f}" fill="none" '
                            f'stroke="#c62828" stroke-width="0.4"/>'
                            f'<path d="M{c[0] - 2:.2f},{c[1]:.2f} H{c[0] + 2:.2f} M{c[0]:.2f},{c[1] - 2:.2f} V{c[1] + 2:.2f}" '
                            f'stroke="#c62828" stroke-width="0.3"/>')
        return self._svg(body)


def fit_model(config: Path, head_name: str | None = None, compare: bool = False) -> list[FitResult]:
    """Fit the YAML's head (or `head_name`) and write the outputs; `compare` also fits every preset."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    fit = dict(raw.get("fit") or {})
    if head_name:
        fit = {k: v for k, v in fit.items() if k not in ("circumference_mm", "eye_distance_mm")} | {"head": head_name}
    folder = config.parent
    data = json.loads((folder / "fold3d.json").read_text(encoding="utf-8"))
    front = fit.get("front", "eyes")
    mask = FoldedMask(data, _axis(fit.get("up", "+y")), None if front == "eyes" else _axis(front))
    field_ = ClearanceField(mask)
    clearance = float(fit.get("clearance_mm", 10.0))
    finder_hole = float(fit.get("sight_hole_mm", SIGHT_HOLE_MM))
    wanted = [fit] + ([{"head": n} for n in HEAD_PRESETS] if compare else [])
    results = []
    for k, one in enumerate(wanted):
        head = HeadModel(HeadSize.from_fit(one))
        fitter = HeadFitter(mask, head, clearance, field_)
        holes = mask.eye_openings() if str(fit.get("eyes", "openings")) == "openings" else None
        span = None
        if holes:
            scale, pupils, margin, span = fitter.fit_to_eyes(holes)
        else:
            scale, pupils, margin = fitter.fit()
        size = np.ptp(mask.triangles.reshape(-1, 3), axis=0) * scale
        result = FitResult(head=asdict(head.size), scale=round(scale, 3), clearance_mm=clearance,
                           pupils_mm=pupils.round(1).tolist(), tightest_mm=round(margin, 1),
                           mask_size_mm=size.round(0).tolist(), eyes="openings" if holes else "free",
                           eye_scale_range=[round(v, 3) for v in span] if span else None)
        finder = SightFinder(mask, scale, finder_hole)
        result.sight = [finder.look(p + pupils) for p in head.pupils]
        results.append(result)
        if k == 0:
            (folder / "head-fit.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
            writer = TemplateWriter(mask, data, head, result)
            (folder / "head-fit-views.svg").write_text(writer.views(), encoding="utf-8")
            (folder / "head-fit-eyes.svg").write_text(writer.eyes(), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head", choices=sorted(HEAD_PRESETS), help="fit this preset instead of the YAML's")
    parser.add_argument("--compare", action="store_true", help="also print the fit of every preset")
    args = parser.parse_args()
    for k, result in enumerate(fit_model(args.config, args.head, args.compare)):
        if k == 1:
            print("presets:")
        print(f"{result.head['name']} head: mask at {result.scale:.0%}, "
              f"{result.mask_size_mm[0]:.0f} x {result.mask_size_mm[1]:.0f} x {result.mask_size_mm[2]:.0f} mm, "
              f"tightest {result.tightest_mm} mm")
        for side, sight in zip(("left eye", "right eye"), result.sight):
            where = (f"through {sight['piece']}, {sight['distance_mm']} mm ahead" if sight["through"] == "card"
                     else "out through an opening")
            print(f"  {side}: {where}")


if __name__ == "__main__":
    main()
