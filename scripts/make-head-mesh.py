"""Cut the head, neck and shoulders out of MakeHuman's base mesh (CC0) for headfit.py.

    python scripts/make-head-mesh.py [base.obj]

Without an argument it downloads base.obj from the MakeHuman repository. Writes
assets/head/makehuman-head.obj: millimetres, y up, the face looking along +z, pupils as the
`pupil` group (two points), body faces from the shoulders up, triangulated.
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj"
OUT = Path(__file__).resolve().parent.parent / "assets" / "head" / "makehuman-head.obj"
CUT_BELOW_EYES_MM = 330.0       # keep the body down to the upper chest
EYE_RADIUS_MM = 11.0            # pupil sits this far in front of the eye joint


def read(text: str):
    vertices, groups, group = [], {}, None
    for line in text.splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
            groups[group] = []
        elif line.startswith("f ") and group:
            groups[group].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(vertices) * 100.0, groups          # MakeHuman works in decimetres


def main() -> None:
    text = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else urllib.request.urlopen(URL).read().decode()
    vertices, groups = read(text)
    eyes = [vertices[sorted({i for f in groups[g] for i in f})].mean(0) for g in ("joint-r-eye", "joint-l-eye")]
    pupils = [e + np.array([0.0, 0.0, EYE_RADIUS_MM]) for e in eyes]
    cut = pupils[0][1] - CUT_BELOW_EYES_MM
    faces = [f for f in groups["body"] if all(vertices[i][1] > cut for i in f)]
    used = sorted({i for f in faces for i in f})
    index = {old: new + 1 for new, old in enumerate(used)}
    lines = ["# Head, neck and shoulders of the MakeHuman base mesh (hm08), cut by scripts/make-head-mesh.py.",
             "# The MakeHuman base mesh was released as CC0 in September 2020 by its copyright holders",
             "# (Data Collection AB and the MakeHuman team): http://www.makehumancommunity.org",
             "# Millimetres, y up, face along +z. Group `pupil`: the two pupils (right, left)."]
    lines += [f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in vertices[used]]
    lines += [f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in pupils]
    lines.append("g body")
    for f in faces:
        for k in range(1, len(f) - 1):
            lines.append(f"f {index[f[0]]} {index[f[k]]} {index[f[k + 1]]}")
    n = len(used)
    lines += ["g pupil", f"p {n + 1} {n + 2}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(used)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    main()
