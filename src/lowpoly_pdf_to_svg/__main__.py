"""Convert the pages listed in a project manifest into Inkscape review SVGs."""

from __future__ import annotations

import argparse
import collections
import csv
import math
import datetime
import pickle
import shutil
import sys
from pathlib import Path

from .config import Project
from .checks import TraceChecks
from .corrections import FOLD_MM, Correction, Corrections, LineFixes, PieceFix, PieceFixes
from .detect import PageDetector
from .edges import EdgeTable
from .ocr import EdgeNumberReader, read_paper
from .render import PageRenderer
from .svg import InkscapeSvgWriter
from .symmetry import Symmetry


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace papercraft PDF pages into Cricut-oriented SVGs.")
    parser.add_argument("--config", type=Path, default=Path("data/example/example.yaml"), help="project YAML manifest")
    parser.add_argument("--output", type=Path, help="output directory (default: the YAML's folder)")
    parser.add_argument("--dpi", type=int, default=180, help="PDF render resolution (default: 180)")
    parser.add_argument("--page", type=int, action="append", help="convert only this 1-based page; repeatable")
    parser.add_argument("--redo", type=int, nargs="*", metavar="PAGE",
                        help="trace only these pages again and reuse the saved trace of every other page "
                             "(no pages: just redraw the SVG and redo the edge check)")
    parser.add_argument("--pdftoppm", default="pdftoppm", help="Poppler pdftoppm executable")
    args = parser.parse_args()

    try:
        project = Project.load(args.config)
        pages = project.pages()
    except ValueError as error:
        parser.error(str(error))
    if not project.pdf.is_file():
        parser.error(f"PDF not found: {project.pdf}")
    renderer_path = shutil.which(args.pdftoppm) or (args.pdftoppm if Path(args.pdftoppm).is_file() else None)
    if not renderer_path:
        parser.error("pdftoppm was not found; install Poppler or pass --pdftoppm")
    if args.page:
        missing = set(args.page) - {page for page, _ in pages}
        if missing:
            parser.error(f"page(s) not assigned to a material: {', '.join(map(str, sorted(missing)))}")
        pages = [(page, material) for page, material in pages if page in args.page]

    output = args.output or args.config.parent
    cache = output / "cache"
    saved = {}
    if args.redo is not None:
        saved = {number: cache / f"page-{number:02d}.pickle" for number, _ in pages if number not in args.redo}
        missing = [str(n) for n, path in saved.items() if not path.is_file()]
        if missing:
            parser.error(f"no saved trace for page(s) {', '.join(missing)}; run a full make first")
    renderer = PageRenderer(project.pdf, args.dpi, renderer_path)
    detector = PageDetector(folds=project.folds)
    results, fresh = [], []
    for number, material in pages:
        if number in saved:
            with saved[number].open("rb") as handle:
                page, layout = pickle.load(handle)
            results.append((page, layout, material))
            continue
        stem = f"page-{number:02d}-{material.id}"
        page = renderer.render(number, output / "previews" / f"{stem}.png")
        layout = detector.detect(page, material.id)
        layout.paper = read_paper(layout.footer)
        if layout.paper and layout.paper not in (material.id, material.name.lower()):
            print(f"warning: page {number} says 'print on {layout.paper} paper' but the config says {material.id}",
                  file=sys.stderr)
        results.append((page, layout, material))
        fresh.append(results[-1])
        print(f"page {number:02d} ({material.id}): {len(results[-1][1].pieces)} pieces", flush=True)

    reader = EdgeNumberReader()
    if not reader.available:
        print("warning: tesseract was not found, so edge numbers are left as '?'", file=sys.stderr)
    corrections = Corrections.load(output / f"number-corrections.csv")
    new_labels = [label for _, layout, _ in fresh for piece in layout.pieces for label in piece.labels]
    known = corrections.match([(layout, page.mm_per_px) for page, layout, _ in fresh])
    print(f"reading {len(new_labels)} edge numbers ({len(known)} corrected by hand teach the reader) ...", flush=True)
    reader.read_all(new_labels, known)
    cache.mkdir(parents=True, exist_ok=True)
    for page, layout, _ in fresh:
        with (cache / f"page-{layout.page:02d}.pickle").open("wb") as handle:
            pickle.dump((page, layout), handle)
    if corrections.rows:
        applied, missed = corrections.apply([(layout, page.mm_per_px) for page, layout, _ in results])
        print(f"applied {applied} hand corrections" + (f", {len(missed)} matched no number: "
              + ", ".join(f"{c.piece} {c.read}->{c.number}" for c in missed) if missed else ""))
    # Folds added or dropped and corners moved on the piece review, then every piece tidied (near points
    # merged, overlapping lines split); numbers on changed pieces look for their line again.
    piece_fixes = PieceFixes.load(output / f"piece-edits.csv")
    page_of = {piece.id: page for page, layout, _ in results for piece in layout.pieces}
    changed, missed = piece_fixes.apply([(layout, page.mm_per_px) for page, layout, _ in results],
                                        lambda piece, a, b: detector.fold_kind(page_of[piece], a, b))
    # Corners the folded 3D model says are slightly off (written by the 3D thread); a corner Helga
    # moved by hand keeps her position.
    nudges = PieceFixes.load(output / f"fold3d-moves.csv")
    if nudges.rows:
        hand = [(r.piece, p) for r in piece_fixes.rows if r.action == "move" for p in (r.a, r.b)]
        nudges.rows = [r for r in nudges.rows
                       if not any(piece == r.piece and math.dist(p, r.a) <= FOLD_MM for piece, p in hand)]
        nudged, unmatched = nudges.apply([(layout, page.mm_per_px) for page, layout, _ in results])
        changed |= nudged
        print(f"moved {len(nudges.rows) - len(unmatched)} corners on {len(nudged)} pieces from the 3D fold"
              + (f", {len(unmatched)} matched no corner" if unmatched else ""))
    mirrored, notes = mirror_fixes(results, piece_fixes, detector, page_of, output)
    changed |= mirrored
    _write_rows(output / "mirror-fixes.csv", notes)
    print(f"mirror twins: " + ", ".join(f"{n} {a}" for a, n in
                                        collections.Counter(r["action"] for r in notes).most_common())
          + f" on {len(mirrored)} pieces (mirror-fixes.csv)" if notes else "mirror twins: nothing to fix")
    for page, layout, _ in results:
        for piece in layout.pieces:
            if piece.id in changed:
                for label in piece.labels:
                    detector._mark_edge(label, piece, 1 / page.mm_per_px)
    print(f"changed {len(changed)} pieces by hand or from the 3D fold (folds, corners)" + (f", {len(missed)} rows matched no fold or corner" if missed else ""))
    lines = LineFixes.load(output / f"number-lines.csv")
    if lines.rows:
        applied, missed = lines.apply([(layout, page.mm_per_px) for page, layout, _ in results])
        print(f"applied {applied} hand-drawn lines" + (f", {len(missed)} matched no number" if missed else ""))
    # Clear-cut misreads are fixed without asking; the rest wait for the number check page.
    measured = [(layout, page.mm_per_px) for page, layout, _ in results]
    first = EdgeTable()
    for layout, mm in measured:
        first.add(layout, mm)
    sure = [m for m in first.misreads() if m["sure"] == "yes"]
    Corrections([Correction(m["piece"], float(m["x_mm"]), float(m["y_mm"]), m["read"], m["suggested"])
                 for m in sure], by_hand=False).apply(measured)
    print(f"fixed {len(sure)} clear-cut misreads without asking")
    # Mirror twins predict numbers from the clean pairs on the other side of the model.
    symmetry = Symmetry(measured)
    second = EdgeTable()
    for layout, mm in measured:
        second.add(layout, mm)
    predicted = symmetry.predictions(second.records())
    mirrored = [m for m in predicted if m["sure"] == "yes"]
    Corrections([Correction(m["piece"], float(m["x_mm"]), float(m["y_mm"]), m["read"], m["suggested"])
                 for m in mirrored], by_hand=False).apply(measured)
    print(f"{len(symmetry.twins)} mirror twins; fixed {len(mirrored)} numbers from their twins without asking")
    sure += mirrored
    labels = [label for _, layout, _ in results for piece in layout.pieces for label in piece.labels]

    writer = InkscapeSvgWriter(f"{project.name.title()} papercraft")
    for page, layout, material in results:
        writer.add(page, layout, material)
    folds = [f for _, layout, _ in results for p in layout.pieces for f in p.folds]
    flagged = [p.id for _, layout, _ in results for p in layout.pieces if p.needs_review]
    unsure = sum(not label.trusted for label in labels)
    print(f"{len(folds)} folds ({sum(not f.snapped for f in folds)} stretched far to reach the cut), "
          f"{len(labels)} numbers ({unsure} uncertain).")
    if flagged:
        print(f"check: {', '.join(flagged)}")
    table = EdgeTable()
    for layout, mm in measured:
        table.add(layout, mm)
    checks = TraceChecks(measured)
    checks.scale(table.records())
    table.checks = checks
    found = checks.write(output / f"trace-faults.csv")
    print(f"Wrote {output / 'trace-faults.csv'}: "
          + (", ".join(f"{n} {c}" for c, n in found.most_common()) or "no trace faults"))
    edges = output / f"edge-numbers.csv"
    statuses = table.write(edges)
    records = table.records()
    asked = symmetry.predictions(records)
    misreads = table.write_misreads(output / f"number-misreads.csv",
                                    applied=[dict(m, sure="applied") for m in sure], extra=asked)
    mirrors = output / f"mirror-mismatches.csv"
    off = symmetry.write(mirrors, records)
    print(f"Wrote {mirrors}: {off} lines differ in length from their mirror line")
    # a fold Helga removed by hand is not suggested again from the twin
    dropped = [(r.piece, r.a, r.b) for r in piece_fixes.rows if r.action == "drop"]
    undone = lambda g: any(piece == g["piece"] and min(
        math.dist(a, (g["ax_mm"], g["ay_mm"])) + math.dist(b, (g["bx_mm"], g["by_mm"])),
        math.dist(b, (g["ax_mm"], g["ay_mm"])) + math.dist(a, (g["bx_mm"], g["by_mm"]))) <= 4.0
        for piece, a, b in dropped)
    gaps = [g for g in symmetry.fold_gaps() if not (g["issue"] == "missing" and undone(g))]
    with (output / f"mirror-missing-folds.csv").open("w", newline="", encoding="utf-8") as handle:
        table_out = csv.DictWriter(handle, ["piece", "twin", "issue", "kind", "ax_mm", "ay_mm", "bx_mm", "by_mm"],
                                   extrasaction="ignore")
        table_out.writeheader()
        table_out.writerows(gaps)
    missing = [g for g in gaps if g["issue"] == "missing"]
    print(f"Wrote {output / 'mirror-missing-folds.csv'}: {len(missing)} folds whose mirror twin has them "
          f"on {len({g['piece'] for g in missing})} pieces, {len(gaps) - len(missing)} mirror folds in another style")
    print(f"Wrote {output / 'number-misreads.csv'}: {len(misreads)} likely misreads left to check")
    table.write_report(output / "README.md", project.name, datetime.date.today().isoformat(), applied=len(sure),
                       symmetry=(symmetry, off))
    print(f"Wrote {edges}: " + ", ".join(f"{n} {s}" for s, n in statuses.most_common()))
    target = output / f"{project.name}.svg"
    writer.write(target)
    print(f"Wrote {len(pages)} page(s) to {target}. {len(flagged)} piece(s) flagged for review.")
    return 0


def mirror_fixes(results, piece_fixes: PieceFixes, detector: PageDetector, page_of: dict,
                 output: Path) -> tuple[set[str], list[dict]]:
    """Mirror twins have the same faces (tabs may differ). Their face corners move to the mirrored
    mean, a fold one twin lacks is added where the scan shows one, and mirror folds read in
    different styles both take the style the scan shows more clearly. A twin that folded cleanly
    in the last 3D fold is the truth for the other. Corners and folds Helga edited by hand keep
    her version."""
    measured = [(layout, page.mm_per_px) for page, layout, _ in results]
    twins = Symmetry(measured)
    hand_moved = [(r.piece, p) for r in piece_fixes.rows if r.action == "move" for p in (r.a, r.b)]
    dropped = [(r.piece, r.a, r.b) for r in piece_fixes.rows if r.action == "drop"]
    near = lambda piece, ends, edits: any(piece == p and min(
        math.dist(a, ends[0]) + math.dist(b, ends[1]), math.dist(b, ends[0]) + math.dist(a, ends[1])) <= 4.0
        for p, a, b in edits)
    # A piece the last 3D fold flagged (corner >1 mm off, glue open) is not clean; its twin may be.
    inspect = output / "fold3d-inspect.csv"
    flagged = {r["piece"] for r in csv.DictReader(inspect.open(encoding="utf-8"))} if inspect.is_file() else None
    clean = {piece.id for _, layout, _ in results for piece in layout.pieces} - flagged if flagged is not None else set()
    rows = [r for r in twins.corner_moves(clean)
            if not any(piece == r["piece"] and math.dist(p, (r["ax_mm"], r["ay_mm"])) <= FOLD_MM for piece, p in hand_moved)]
    pieces = {piece.id: piece for _, layout, _ in results for piece in layout.pieces}
    px = lambda piece, point: (point[0] / page_of[piece].mm_per_px, point[1] / page_of[piece].mm_per_px)
    restyled = set()
    for gap in twins.fold_gaps():
        ends = ((gap["ax_mm"], gap["ay_mm"]), (gap["bx_mm"], gap["by_mm"]))
        if near(gap["piece"], ends, dropped):
            continue
        a, b = px(gap["piece"], ends[0]), px(gap["piece"], ends[1])
        if gap["issue"] == "missing":
            if detector.fold_kind(page_of[gap["piece"]], a, b) is not None:
                rows.append({"piece": gap["piece"], "action": "add", "kind": gap["kind"],
                             "ax_mm": ends[0][0], "ay_mm": ends[0][1], "bx_mm": ends[1][0], "by_mm": ends[1][1],
                             "note": f"fold of mirror twin {gap['twin']}, printed on the scan"})
            continue
        ta, tb = (px(gap["twin"], tuple(p)) for p in gap["twin_fold"])
        here = detector.fold_evidence(page_of[gap["piece"]], a, b)
        there = detector.fold_evidence(page_of[gap["twin"]], ta, tb)
        if here is None or there is None:
            continue
        clearer = here if abs(here) >= abs(there) else there
        kind = detector.folds["dash_dot" if clearer >= 0 else "dashed"]
        for piece_id, p, q in ((gap["piece"], a, b), (gap["twin"], ta, tb)):
            folds = pieces[piece_id].folds
            if not folds:
                continue
            fold = min(folds, key=lambda f: min(math.dist(f.a, p) + math.dist(f.b, q), math.dist(f.a, q) + math.dist(f.b, p)))
            if fold.kind != kind:
                fold.kind = kind
                restyled.add(piece_id)
                mm = page_of[piece_id].mm_per_px
                rows.append({"piece": piece_id, "action": "style", "kind": kind.value,
                             "ax_mm": round(fold.a[0] * mm, 2), "ay_mm": round(fold.a[1] * mm, 2),
                             "bx_mm": round(fold.b[0] * mm, 2), "by_mm": round(fold.b[1] * mm, 2),
                             "note": f"mirror fold on {gap['twin'] if piece_id == gap['piece'] else gap['piece']} "
                                     f"read differently; the scan reads {kind.value} more clearly"})
    edits = PieceFixes([PieceFix(r["piece"], r["action"], r["kind"], (r["ax_mm"], r["ay_mm"]), (r["bx_mm"], r["by_mm"]))
                        for r in rows if r["action"] in ("move", "add")])
    changed, _ = edits.apply(measured, lambda piece, a, b: detector.fold_kind(page_of[piece], a, b))
    return changed | restyled, rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["piece", "action", "kind", "ax_mm", "ay_mm", "bx_mm", "by_mm", "note"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
