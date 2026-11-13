"""Render a folded fold3d mesh (OBJ + MTL, millimetres) as flat-shaded views for the paper.

Usage: python3 render_mask.py fold3d.obj out.png
Only our own reconstruction is drawn; no template pages or scans are used.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class FoldedMesh:
    def __init__(self, obj: Path):
        self.colour = self._materials(obj.with_suffix(".mtl"))
        self.vertices, self.faces, self.materials = [], [], []
        material = None
        for line in obj.read_text().splitlines():
            part = line.split()
            if not part:
                continue
            if part[0] == "v":
                self.vertices.append([float(x) for x in part[1:4]])
            elif part[0] == "usemtl":
                material = part[1]
            elif part[0] == "f":
                self.faces.append([int(p.split("/")[0]) - 1 for p in part[1:]])
                self.materials.append(material)
        self.vertices = np.array(self.vertices)

    def pitch(self, degrees: float) -> None:
        """Tilt the snout (+z) up towards +y, so the mask sits as it would on a head."""
        t = np.radians(degrees)
        y, z = self.vertices[:, 1].copy(), self.vertices[:, 2].copy()
        self.vertices[:, 1] = y * np.cos(t) + z * np.sin(t)
        self.vertices[:, 2] = -y * np.sin(t) + z * np.cos(t)

    @staticmethod
    def _materials(mtl: Path) -> dict:
        colour, name = {}, None
        for line in mtl.read_text().splitlines():
            part = line.split()
            if part[:1] == ["newmtl"]:
                name = part[1]
            elif part[:1] == ["Kd"]:
                colour[name] = np.array([float(x) for x in part[1:4]])
        return colour

    def draw(self, ax, elev: float, azim: float, light=np.array([0.4, -0.6, 0.7])):
        polys = [self.vertices[f] for f in self.faces]
        light = light / np.linalg.norm(light)
        colours = []
        for poly, m in zip(polys, self.materials):
            n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
            n = n / (np.linalg.norm(n) or 1)
            shade = 0.45 + 0.55 * abs(n @ light)  # two-sided: inside faces lit too
            colours.append(np.clip(self.colour.get(m, np.full(3, 0.7)) * shade, 0, 1))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors=(0, 0, 0, 0.25),
                                             linewidths=0.15))
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        mid, r = (lo + hi) / 2, (hi - lo).max() / 2
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setter(c - r, c + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim, vertical_axis="y")
        ax.set_axis_off()


def main(obj: str, out: str, views=((5, 0), (8, 40), (0, 90)), gap: int = 30, pitch: float = 40):
    """One render per view, each cropped to the model, packed side by side with a small gap.
    View (5, 0) looks at the face (the mesh faces +z, y up)."""
    from PIL import Image, ImageChops
    import io
    mesh = FoldedMesh(Path(obj))
    mesh.pitch(pitch)  # worn pose: face upright, snout forward
    tiles = []
    for elev, azim in views:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection="3d")
        mesh.draw(ax, elev, azim)
        fig.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=300)
        plt.close(fig)
        tile = Image.open(buf).convert("RGB")
        box = ImageChops.difference(tile, Image.new("RGB", tile.size, "white")).getbbox()
        tiles.append(tile.crop(box))
    height = max(t.height for t in tiles)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), height), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, height - t.height))
        x += t.width + gap
    sheet.save(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
