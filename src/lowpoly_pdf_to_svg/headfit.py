"""Fit the folded mask to a wearer's head and print where to cut the sight holes (issue #79).

Reads the folded model (data/<m>/fold3d.json, millimetres) and the YAML `fit:` block:

    fit:
      head: female            # child | female | male preset
      circumference_mm: 545   # optional, scales the preset
      eye_distance_mm: 61     # optional, pupil to pupil
      clearance_mm: 0         # air between head and card (it rests on the head)
      sight_hole_mm: 30       # diameter of each sight hole
      eyes: openings          # pupils centred behind the model's eye openings (or: free)
      up: +y                  # roughly up in the model's axes
      front: eyes             # the way the eye openings face (or an axis such as -y)

The head is MakeHuman's base mesh (CC0) from the neck up, scaled to the head's circumference. With
`eyes: openings` (the default, when the model has a mirrored pair of openings in front) the mask
faces the way its eye openings face and gets the smallest scale at which the head fits with each
pupil inside its opening. With `eyes: free` the mask is scaled until the head, with clearance, just
fits anywhere. Head that leaves through the neck opening (the largest one) is allowed either way.

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
from xml.sax.saxutils import escape

# Rounded adult means in the style of ANSUR II (2012) and a ~8 year old child; mm. They are a
# starting point: a measured head circumference and pupil distance always win.
#   breadth: ear to ear above the ears, length: brow to back of head, height: chin to crown.
HEAD_PRESETS = {
#   circumference: round the head just above the brow and ears, as a tape measure gives it.
    "male":   dict(breadth=154.0, length=199.0, height=236.0, circumference=572.0, eye_distance=64.0, stature=1804.0),
    "female": dict(breadth=146.0, length=187.0, height=221.0, circumference=549.0, eye_distance=62.0, stature=1676.0),
    "child":  dict(breadth=140.0, length=177.0, height=200.0, circumference=525.0, eye_distance=56.0, stature=1305.0),
}
# stature: body height in mm. Adults: Swedish 18-year-olds; child: Swedish 8-year-olds (Albertsson
# Wikland et al., Acta Paediatr 2002;91:739-754). It sizes the shoulders and chest (`height_cm`).

CHIN_BELOW_PUPILS = -0.49     # chin this far below the pupils, as a share of the head height
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
    stature: float = 1676.0
    name: str = "custom"
    body_scale: float = 1.0      # shoulders and chest relative to the head's own scale

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
        if "height_cm" in fit:
            # The body scales with height; the head (circumference) much less, so the two are separate.
            stature = 10.0 * float(fit["height_cm"])
            preset = HEAD_PRESETS[name]
            body = (stature / preset["stature"]) / (head.circumference / preset["circumference"])
            head = replace(head, stature=stature, body_scale=body, name=f"{head.name}, {stature / 10:.0f} cm")
        return head


HEAD_MESH = Path(__file__).resolve().parents[2] / "assets" / "head" / "makehuman-head.obj"
HEAD_CREDIT = ("MakeHuman base mesh, Data Collection AB and the MakeHuman team (CC0), "
               "makehumancommunity.org")
CIRCUMFERENCE_ABOVE_PUPILS_MM = 35.0     # where a tape goes round the head, above the brows
NECK_BELOW_CHIN_MM = 50.0                # how much neck is kept below the chin
TORSO_GIVE_MM = 5.0                      # the card may press this far into clothes and skin


class HeadModel:
    """A real head shape: the MakeHuman base mesh (CC0) from the shoulders up, scaled to the head size.

    Head frame: x to the wearer's right, y back, z up, origin between the pupils, millimetres. The
    mesh is scaled evenly so its circumference above the brows matches the size; a given eye
    distance moves the pupil marks only.
    """

    def __init__(self, size: HeadSize, mesh: Path = HEAD_MESH):
        self.size = size
        vertices, faces, pupils = self._read(mesh)
        # MakeHuman: x to the person's left, y up, face along +z
        turn = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        vertices, pupils = vertices @ turn.T, pupils @ turn.T
        origin = pupils.mean(0)
        vertices, pupils = vertices - origin, pupils - origin
        k = size.circumference / self._circumference(vertices)
        self.vertices = vertices * k
        # Head and neck must fit inside; the shoulders and chest below must not be cut by the card
        # (the mask rests on them, and they stop the head going further in).
        neck = CHIN_BELOW_PUPILS * size.height - NECK_BELOW_CHIN_MM
        if size.body_scale != 1.0:                  # shoulders and chest sized by body height
            low = self.vertices[:, 2] < neck
            ring = self.vertices[np.abs(self.vertices[:, 2] - neck) < 10.0]
            pivot = np.array([0.0, ring[:, 1].mean() if len(ring) else 0.0, neck])
            self.vertices[low] = pivot + (self.vertices[low] - pivot) * size.body_scale
        upper = (self.vertices[faces][:, :, 2] > neck).all(axis=1)
        self.faces, self.torso_faces = faces[upper], faces[~upper]
        self.triangles = self.vertices[self.faces]
        self.torso_triangles = self.vertices[self.torso_faces]
        self._normals()
        half = size.eye_distance / 2
        self.pupils = np.array([[-half, pupils[0, 1] * k, 0.0], [half, pupils[1, 1] * k, 0.0]])
        crown = self.vertices[:, 2].max()
        self.core_centre = np.array([0.0, 0.25 * size.length, crown - 0.5 * size.breadth])
        self.core_radius = 0.5 * size.breadth

    @staticmethod
    def _read(path: Path):
        vertices, faces, points = [], [], []
        for line in path.read_text().splitlines():
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
            elif line.startswith("p "):
                points = [int(t) - 1 for t in line.split()[1:]]
        vertices = np.array(vertices)
        return vertices, np.array(faces), vertices[points]

    @staticmethod
    def _circumference(vertices: np.ndarray) -> float:
        ring = vertices[np.abs(vertices[:, 2] - CIRCUMFERENCE_ABOVE_PUPILS_MM) < 2.0][:, :2]
        hull = _convex_hull(ring)
        return float(np.linalg.norm(hull - np.roll(hull, 1, axis=0), axis=1).sum())

    def _normals(self) -> None:
        """Outward vertex normals of the torso, for the inside test."""
        t = self.vertices[self.torso_faces]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        normals = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(normals, self.torso_faces[:, k], n)
        centre = self.vertices[self.torso_faces].reshape(-1, 3).mean(0)
        used = np.unique(self.torso_faces)
        outward = ((self.vertices[used] - centre) * normals[used]).sum(1).mean()
        if outward < 0:
            normals = -normals
        length = np.linalg.norm(normals, axis=1)
        self.torso_points = self.vertices[used]
        self.torso_normals = normals[used] / length[used, None]

    def torso_depth(self, points: np.ndarray) -> np.ndarray:
        """How far each point (head frame) is inside the shoulders and chest, mm; 0 outside."""
        from scipy.spatial import cKDTree
        if getattr(self, "_tree", None) is None:
            self._tree = cKDTree(self.torso_points)
        distance, index = self._tree.query(points, distance_upper_bound=60.0)
        inside = np.zeros(len(points))
        ok = np.isfinite(distance)
        side = ((points[ok] - self.torso_points[index[ok]]) * self.torso_normals[index[ok]]).sum(1)
        inside[ok] = np.where(side < 0, distance[ok], 0.0)
        return inside

    def tip(self, degrees: float) -> None:
        """Tip the head about the line between the pupils."""
        turn = _pitch(degrees)
        self.vertices = self.vertices @ turn.T
        self.triangles = self.vertices[self.faces]
        self.core_centre = turn @ self.core_centre
        self.torso_points = self.torso_points @ turn.T
        self.torso_normals = self.torso_normals @ turn.T
        self._tree = None
        self.torso_triangles = self.vertices[self.torso_faces]

    def samples(self, spacing: float = 8.0) -> np.ndarray:
        """Points on the skin from the shoulders up, about `spacing` mm apart."""
        t = self.triangles
        area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        count = int(area.sum() / spacing ** 2)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(t), size=count, p=area / area.sum())
        u, v = rng.random(count), rng.random(count)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        tri = t[pick]
        return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])


def _pitch(degrees: float) -> np.ndarray:
    """Rotation about the ear-to-ear axis; positive tips the face up."""
    a = math.radians(degrees)
    return np.array([[1.0, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


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

    def neck_cap(self) -> np.ndarray:
        """The neck opening closed by a fan of triangles round its centre (for the through-test)."""
        if getattr(self, "_cap", None) is None:
            loop = self.openings[0]
            c = loop.mean(0)
            u, v = np.linalg.svd(loop - c)[2][:2]
            ring = loop[np.argsort(np.arctan2((loop - c) @ v, (loop - c) @ u))]
            self._cap = np.array([[c, a, b] for a, b in zip(ring, np.roll(ring, -1, axis=0))])
        return self._cap

    def beyond_neck(self, points: np.ndarray, centre: np.ndarray | None = None) -> np.ndarray:
        """Points that have left the mask through the neck opening: the line from `centre` (inside
        the head) to the point crosses the opening. Without a centre, the side of its plane."""
        if centre is None:
            return (points - self.neck_centre) @ self.neck_normal > 0
        cap = self.neck_cap()
        d = points - centre                                   # segments centre -> point
        e1, e2 = cap[:, 1] - cap[:, 0], cap[:, 2] - cap[:, 0]
        h = np.cross(d[:, None, :], e2[None])
        det = (e1[None] * h).sum(2)
        inv = 1.0 / np.where(np.abs(det) < 1e-9, np.inf, det)
        sv = centre - cap[:, 0]
        uu = (sv[None] * h).sum(2) * inv
        q = np.cross(sv, e1)
        vv = (d @ q.T) * inv
        t = (q * e2).sum(1)[None] * inv
        hit = (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t > 0) & (t < 1)
        return hit.any(axis=1)

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
    tip_deg: float = 0.0               # head tipped this far (+ face up) to fit
    rule: str = "eyes"                 # "inside": smallest scale with the whole head inside the card
    rests_on: str = ""                 # "crown" or "shoulders" for the "inside" rule
    eye_scale: float | None = None     # scale at which the eye openings sit over the pupils
    eye_offset_mm: list | None = None  # pupils from the eye openings' centre: [behind, up] in mm
    look_up_deg: list | None = None    # gaze tilt up to see out: [to the opening's lower edge, to its centre]
    sight: list = field(default_factory=list)


class HeadFitter:
    """Smallest mask scale at which the head, with clearance, fits inside the card."""

    def __init__(self, mask: FoldedMask, head: HeadModel, clearance: float, field: "ClearanceField | None" = None,
                 spacing: float = 8.0):
        self.mask, self.head, self.clearance = mask, head, clearance
        self.field = field or ClearanceField(mask)
        self.samples = head.samples(spacing)

    def torso_cut(self, scale: float, pupils: np.ndarray, tip: float) -> float:
        """Deepest the card goes into the shoulders and chest, mm."""
        if getattr(self, "_card", None) is None:
            self._card = self.mask.surface_points(spacing=6.0)
        points = (self._card * scale - pupils) @ _pitch(tip)       # into the upright head's frame
        return float(self.head.torso_depth(points).max())

    def margin(self, scale: float, pupils: np.ndarray, chin_up: bool = False) -> float:
        """Least air between head and card in real mm, ignoring head that has left through the neck.
        `chin_up`: only the head from the chin up; the neck and shoulders may pass the card's rim."""
        samples = self.samples[self.samples[:, 2] >= CHIN_BELOW_PUPILS * self.head.size.height] if chin_up else self.samples
        points = (samples + pupils) / scale
        keep = ~self.mask.beyond_neck(points, (self.head.core_centre + pupils) / scale)
        return float((scale * self.field(points[keep])).min()) if keep.any() else -1e3

    def best_position(self, scale: float, step: float = 4.0) -> tuple[np.ndarray, float]:
        low, high = self.field.low * scale, self.field.high * scale
        ys, zs = np.arange(low[1], high[1], step), np.arange(low[2], high[2], step)
        grid = np.array([[0.0, y, z] for y in ys for z in zs])
        # the cranium centre must be at least its smallest radius plus clearance from the card
        centre = grid + self.head.core_centre
        need = (self.head.core_radius + self.clearance) * 0.9
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

    def fit_to_eyes(self, holes: tuple[np.ndarray, np.ndarray], edge_mm: float = 5.0,
                    tolerance: float = 0.01) -> tuple[float, np.ndarray, float, tuple, float]:
        """Pupils behind the eye openings, each at least `edge_mm` inside its opening: the smallest
        scale at which the head, tipped forward or back up to 30 degrees, keeps `clearance` from the
        card. Falls back to the largest scale the eyes allow. Returns scale, pupil midpoint, margin,
        the scale range the eye distance allows and the head's tip (degrees, + looks up)."""
        centres = np.array([h.mean(0) for h in holes])
        half_gap = (centres[1, 0] - centres[0, 0]) / 2
        half_width = np.mean([np.ptp(h[:, 0]) for h in holes]) / 2
        ipd = self.head.size.eye_distance / 2
        lowest = (ipd + edge_mm) / (half_gap + half_width)
        highest = (ipd - edge_mm) / (half_gap - half_width) if half_gap > half_width else 2.5
        mid = np.array([0.0, centres[:, 1].mean(), centres[:, 2].mean()])
        upright = self.samples

        def best(scale):
            """Best pose: most head room among poses whose card stays out of the chest."""
            top = None
            for tip in (0,):                                     # upright: Helga, 2026-11-04
                self.samples = upright @ _pitch(tip).T
                for relief in (10.0, 15.0, 20.0):                # face close behind the eye openings
                    for dz in (-8.0, 0.0, 8.0):                  # a little up or down in the opening
                        p = mid * scale + np.array([0.0, relief, dz])
                        m = self.margin(scale, p)
                        if top is not None and m <= top[1]:
                            continue
                        if self.torso_cut(scale, p, tip) > TORSO_GIVE_MM:
                            continue
                        top = (p, m, tip)
            self.samples = upright
            if top is None:                                      # every pose cuts the chest: least cut
                self.samples = upright
                cuts = [(self.torso_cut(scale, mid * scale + np.array([0.0, r, dz]), 0), r, dz)
                        for r in (10.0, 15.0, 20.0) for dz in (-8.0, 0.0, 8.0)]
                _, r, dz = min(cuts)
                p = mid * scale + np.array([0.0, r, dz])
                top = (p, self.margin(scale, p), 0)
            return top
        # The eye openings sit centred over the pupils; the head stands upright and the mask rests on
        # the shoulders. The head room inside is reported as a check: the fold is rough inside.
        scale = ipd / half_gap
        p, m, tip = best(scale)
        return scale, p, m, (lowest, highest), tip

    def fit_inside(self, holes: tuple[np.ndarray, np.ndarray], step: float = 0.02, high: float = 1.6,
                   depth=range(-10, 61, 5), drop=range(40, -121, -5),
                   at: float | None = None) -> tuple[float, np.ndarray, float, str]:
        """Smallest scale, from the eye scale up, at which the whole head stays inside the card
        (head room >= clearance, chin to crown: the neck may pass the rim, and the crown may go up
        into a hollow such as Bowser's hair) with the card at most TORSO_GIVE_MM into the shoulders. The mask
        rests as low as it can: on the crown or on the shoulders, whichever it meets first
        (Helga, 2026-11-04). `at`: hang the mask at this scale instead of searching. Returns scale, pupil
        midpoint, head room and what the mask rests on."""
        centres = np.array([h.mean(0) for h in holes])
        mid = np.array([0.0, centres[:, 1].mean(), centres[:, 2].mean()])
        scale = self.head.size.eye_distance / 2 / ((centres[1, 0] - centres[0, 0]) / 2)
        if at is not None:
            scale = high = float(at)

        def rest(scale):
            top = None
            for r in depth:                      # face this far behind the eye openings
                for dz in drop:                  # head this far above (+) or below the openings
                    if top is not None and dz <= top[2]:
                        break
                    p = mid * scale + np.array([0.0, r, dz])
                    if self.torso_cut(scale, p, 0) > TORSO_GIVE_MM:
                        continue
                    m = self.margin(scale, p, chin_up=True)
                    if m >= self.clearance:
                        top = (p, m, dz)
                        break
            return top
        while scale <= high:
            top = rest(scale)
            if top is not None:
                p, m, _ = top
                up = p + np.array([0.0, 0.0, 5.0])
                on = "shoulders" if self.torso_cut(scale, up, 0) > TORSO_GIVE_MM else "crown"
                return scale, p, m, on
            scale += step
        raise RuntimeError(f"the head does not fit inside even at {high:.0%}")

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

    def __init__(self, mask: FoldedMask, data: dict, head: HeadModel, result: FitResult, credit: str | None = None):
        self.mask, self.data, self.head, self.result = mask, data, head, result
        self.credit = credit

    def credits(self) -> list[str]:
        """Who made what is drawn: the mask's template (YAML source.credit) and the head and torso."""
        return ([f"Mask: {self.credit}."] if self.credit else []) + ["Head and torso: " + HEAD_CREDIT + "."]

    def _svg(self, body: list[str]) -> str:
        w, h = A4_MM
        lines = self.credits()
        top = h - 4 - 3.6 * len(lines)
        body = body + [f'<rect x="0" y="{top:g}" width="{w}" height="{h - top:g}" fill="white" opacity="0.85"/>'] + [
            f'<text x="12" y="{top + 3.6 * (k + 1):.1f}" font-size="2.8" fill="#555">{escape(line)}</text>'
            for k, line in enumerate(lines)]
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
                + (f'(head in, {r.tightest_mm:.0f} mm room, on {r.rests_on})</text>'
                 if r.rule == "inside" else
                 f'(head room check: {r.tightest_mm:.0f} mm, wanted {r.clearance_mm:g})</text>'),
                f'<text x="12" y="20" font-size="3.2">Mask {r.mask_size_mm[0]:.0f} × {r.mask_size_mm[1]:.0f} × '
                f'{r.mask_size_mm[2]:.0f} mm (wide × deep × tall). Grey: card. Blue: head (MakeHuman, CC0). Red: pupils and sight.</text>']
        if r.eye_offset_mm:
            back, up = r.eye_offset_mm
            look = (f' Looking up {r.look_up_deg[0]:.0f}° reaches the openings (centre {r.look_up_deg[1]:.0f}°).'
                    if r.look_up_deg else "")
            body.append(f'<text x="12" y="25" font-size="3.2">Pupils {back:.0f} mm behind and {abs(up):.0f} mm '
                        f'{"below" if up < 0 else "above"} the centre of the eye openings.{look}</text>')
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
            chest = self.head.torso_triangles + head_offset
            body.append('<g fill="#7a9cc6" stroke="#7a9cc6" stroke-width="0.2" opacity="0.18">')
            for tri_ in chest:
                x, y = xy(tri_)
                if np.all(np.abs(x - cx) < 100) and np.all(np.abs(y - cy) < 66):
                    body.append('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + 'Z"/>')
            body.append("</g>")
            skin = self.head.triangles + head_offset
            order = np.argsort(skin[:, :, depth].mean(1) * (1 if name == "front" else -1))
            body.append('<g fill="#1f5fa8" stroke="#1f5fa8" stroke-width="0.2" opacity="0.3">')
            for i in order:
                x, y = xy(skin[i])
                body.append('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + 'Z"/>')
            body.append("</g>")
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
    clearance = float(fit.get("clearance_mm", 0.0))
    finder_hole = float(fit.get("sight_hole_mm", SIGHT_HOLE_MM))
    wanted = [fit] + ([{"head": n} for n in HEAD_PRESETS] if compare else [])
    results = []
    for k, one in enumerate(wanted):
        head = HeadModel(HeadSize.from_fit(one))
        fitter = HeadFitter(mask, head, clearance, field_)
        holes = mask.eye_openings() if str(fit.get("eyes", "openings")) == "openings" else None
        span = None
        tip = 0.0
        rule, rests_on, eye_scale, eye_offset, look_up = str(fit.get("rule", "inside")), "", None, None, None
        if holes and rule == "inside":
            eye_scale, *_, span, _ = fitter.fit_to_eyes(holes)
            scale, pupils, margin, rests_on = fitter.fit_inside(holes, at=fit.get("scale") if k == 0 else None)
            opening = np.array([h.mean(0) for h in holes]).mean(0) * scale
            eye_offset = [round(float(pupils[1] - opening[1]), 1), round(float(pupils[2] - opening[2]), 1)]
            look_up = look_up_degrees(holes, scale, head.pupils + pupils)
        elif holes:
            scale, pupils, margin, span, tip = fitter.fit_to_eyes(holes)
            head.tip(tip)
        else:
            scale, pupils, margin = fitter.fit()
        size = np.ptp(mask.triangles.reshape(-1, 3), axis=0) * scale
        result = FitResult(head=asdict(head.size), scale=round(scale, 3), clearance_mm=clearance,
                           pupils_mm=pupils.round(1).tolist(), tightest_mm=round(margin, 1),
                           mask_size_mm=size.round(0).tolist(), eyes="openings" if holes else "free",
                           eye_scale_range=[round(v, 3) for v in span] if span else None,
                           tip_deg=tip, rule=rule if holes else "free", rests_on=rests_on,
                           eye_scale=round(eye_scale, 3) if eye_scale else None, eye_offset_mm=eye_offset, look_up_deg=look_up)
        finder = SightFinder(mask, scale, finder_hole)
        result.sight = [finder.look(p + pupils) for p in head.pupils]
        results.append(result)
        if k == 0:
            (folder / "head-fit.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
            writer = TemplateWriter(mask, data, head, result, (raw.get("source") or {}).get("credit"))
            (folder / "head-fit-views.svg").write_text(writer.views(), encoding="utf-8")
            (folder / "head-fit-eyes.svg").write_text(writer.eyes(), encoding="utf-8")
            (folder / "head-fit-mesh.json").write_text(json.dumps(head_mesh(mask, head, result)), encoding="utf-8")
    return results


def look_up_degrees(holes, scale: float, pupils: np.ndarray) -> list:
    """How far each eye must look up (degrees, + up) to see out of its eye opening: to the opening's
    lowest point and to its centre; the larger (worse) eye is reported."""
    lower, centre = [], []
    for pupil in pupils:
        hole = min(holes, key=lambda h: abs(h.mean(0)[0] * scale - pupil[0])) * scale
        def up(p):
            ahead = max(pupil[1] - p[1], 1e-6)          # the face looks along -y
            return math.degrees(math.atan2(p[2] - pupil[2], ahead))
        lower.append(up(hole[np.argmin(hole[:, 2])]))
        centre.append(up(hole.mean(0)))
    return [round(max(lower), 1), round(max(centre), 1)]


def head_mesh(mask: FoldedMask, head: HeadModel, result: FitResult) -> dict:
    """The fitted head and shoulders in fold3d.json's axes and size (the mask unscaled, so the head
    shrinks by 1/scale), for the 3D viewer."""
    pupils = np.array(result.pupils_mm)
    vertices = ((head.vertices + pupils) / result.scale) @ mask.frame
    faces = np.vstack([head.faces, head.torso_faces])
    used = np.unique(faces)
    index = np.full(len(vertices), -1)
    index[used] = np.arange(len(used))
    return {"source": "MakeHuman base mesh (CC0), assets/head/makehuman-head.obj", "credit": HEAD_CREDIT,
            "head": result.head["name"], "scale": result.scale,
            "vertices": np.round(vertices[used], 1).tolist(), "faces": index[faces].tolist(),
            "torso_from": int(len(head.faces))}


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
