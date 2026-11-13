# Independent review: EG 2027 short paper draft (issue #97)

Reviewed: `paper/eg2027/` on PR #98 (branch `claude/project-thread-2nma6w`, commit `ca798ef`),
compiled draft `eg2027-short-paper-draft.pdf` (4 pages, built 2026-11-13).
Code: `src/lowpoly_pdf_to_svg/` at the same commit. Data: private repo `tungufoss/papercraft`
at `38aac50` (`data/bowser/`, `data/bowser-fresh/`, `scripts/compare-*.py`, `scripts/oracle-answers.py`)
and `report/data/bowser/*.csv` here.

The reviewer started cold. I used only the paper, the code and the committed data, and no earlier
discussion. I edited nothing and re-ran nothing: every number below was recomputed from committed
CSV/JSON files.

## Summary verdict

The method description is **faithful to the code** in nearly all of its specifics. All 20+
thresholds in the text and pseudocode match constants in the code. Algorithm 2 (two readers + pair-up)
is an almost line-for-line transcription of `ocr.py`. Algorithms 1 and 3 are fair summaries with a
few simplifications that a careful reviewer could catch (listed below).

The **numbers in the table are copied correctly** from the data. The problems are elsewhere:

1. **The evaluation has no ground truth.** "Right partner", "0.6 mm" and "rebuild the checked result"
   are all measured against the hand-corrected build, which was made with this same pipeline.
   The paper presents them as accuracy.
2. **The "all 754 glued edges close" claim is narrower than it reads.** It is within-cardstock glues
   only, at a 3 mm closing tolerance, with 20 glues pinned by hand. The same fold-up reports 20
   openings against 4 expected and 9 folds bent the wrong way. The paper does not mention these.
3. **Human input is undercounted.** The 1079 "rows" leave out the confirmed joins, glue pins, tab
   overrides, accepted corners and other 3D-session files that the "all answers" build needed.
4. **There is no baseline** in the paper, although the repo has a synthetic benchmark with three
   baselines (`bench/`).
5. **Double-blind and policy risks.** The public repo README names the template, its Sketchfab URL
   and the author's GitHub account. The README and the paper credit different designer names. And,
   per the project's notes on the call, only grammar-level generative-AI help is allowed, while
   this draft was written by AI agents.

None of these needs new research for a short paper. Most are wording fixes plus one or two small tables.

---

## 1. Method: pseudocode vs code

Verdict key: **V** verified (code does what the text says), **P** partly (the gist is right but a
detail differs or is missing), **N** not found.

### Algorithm 1 / Section 3.1 (pieces, polygons, folds): `detect.py`

| Paper claim | Code | Verdict |
|---|---|---|
| Render at 180 dpi with `pdftoppm`, mm via page scale | `__main__.py:31` default `--dpi 180`; `render.py:24,44` | V |
| Components longer than 4.3 mm are cut lines | `DetectorSettings.solid_min_mm = 4.3`; `_masks` uses bounding-box extent `max(w,h)` | V (it is bbox extent, not stroke length) |
| Dark short components are glyphs, rest is dashes | `_masks`: glyph = short & darkest < 110, **or any pixel < 110** (`text_core`); red ink removed first | P: the extra rule (all near-black pixels count as glyph) and the red-ink filter are not mentioned |
| Piece = what a border flood fill cannot reach; each enclosed area seeds its own piece | `_piece_regions` (close 0.7 mm, 4-connected background, seed regions ≥ 30 mm², grow through non-outside) | V |
| `ApproxPoly(contour, 1.5 px)` | `_outlines`: `cv2.approxPolyDP(..., outline_tolerance_px=1.5)` | V |
| Straighten: drop bumps from numbers | `_straighten` (≤ 6 mm long, ≤ 2 mm deep, ≤ 8° bend) | V |
| `Repeat` Huber edge fit + corner intersection + replace ≤ 3 vertices **until no change** | `_outlines` runs `_fit_edges(band=4 px)` then `_follow_cut` **exactly twice** (`for _ in range(2)`); `_follow_cut` itself loops to a fixpoint | P: it is 2 fixed rounds, not "until no change". Also undocumented: `_sharpen` (restores rounded acute corners, 1.2 / 2.0 mm) runs before and after |
| Drop vertices turning < 4°; merge edges < 0.8 mm | `_drop_flat` (`min_corner_deg = 4`), `_merge_close` (`min_edge_mm = 0.8`) | V (code merges first, then drops) |
| Hough on D ∩ R, merge collinear | `_folds`: `HoughLinesP` on dashes in eroded region; `_merge_collinear(gap = 8 mm)` iterated; `_refit` | V |
| Extend to cut, snap to corner within 1.5 mm | `_span_piece`, `snap_mm = 1.5` | V |
| Coverage of the stretched fold ≥ 60 % **dashes** | `_coverage(evidence, ...) < fold_min_ink = 0.6`, where evidence = dilate(**dash ∪ glyph**) | P: number glyphs count as support ("60 % dashes or digits") |
| Style from run lengths (dash-dot has long runs) | `_dash_dot_share`, `DASH_DOT_SHARE = 0.25`, run ≥ 15 px at 180 dpi | V |
| Glyph groups attach to the nearest edge or fold; a number on a fold marks a tab | `_labels`, `_mark_edge` (reach 3 mm, angle penalty) | V |
| (not in paper) footer margin 10 mm excluded; folds stretched > 6 mm flagged; stray ink to Review layer | `detect()` | Fine to omit for 4 pages |

### Algorithm 2 / Section 3.2 (numbers): `ocr.py`

| Paper claim | Code | Verdict |
|---|---|---|
| Tesseract reads one line of digits at 0° and 180° | `_tesseract`: `--psm 7`, digit whitelist, flips (0, 180), conf = min word conf | V |
| Matcher trained on marks with conf ≥ 90 plus hand answers | `TRAIN = 90.0`; `_train` adds `known` with conf = ∞; only 2–3 digit texts | V |
| "trained **on the page itself**" | `__main__.py:84-97`: one `EdgeNumberReader` for **all freshly traced pages** of the template | P: it is trained on the whole template, not per page. Say "on the template itself" |
| Accept if both readers agree or Tesseract conf ≥ 93 | `read_all`: `t_text == options[0][0] or t_conf >= TRUST (93)` | V |
| Undecided by decreasing best score; candidates within 0.05; prefer completing, then starting a pair | `_pair_up` | V, exact |
| Trust only if it completes a pair and was ranked first by one reader | `completes = counts==1 and (pick is options[0] or pick == tesseract)` | V |
| Flag numbers with n(x) ≠ 2 | `label.paired = counts[text] == 2` | V |
| (missing) what the matcher *is* | `_candidates`: nearest-template matching, mean of top-3 cosine similarities per digit, splits into 1/2/3 cells, aspect penalty 0.3 | N in paper. A reviewer will ask; one clause is enough |

### Section 3.3 (constraints): `edges.py`, `symmetry.py`, `checks.py`, `__main__.py`

| Paper claim | Code | Verdict |
|---|---|---|
| Statuses ok / unpaired / length mismatch / no tab / both tabs / extra copy, within 1 mm | `edges.py:18` `LENGTH_TOLERANCE_MM = 1.0`, `_status` | V (code also has "printed N times" and "line not found", which appear in the fresh data: 30 and 32 rows) |
| Unpaired number matched to an unpaired same-length line when it is the reader's alternative or one digit differs, and no other candidate | `misreads()` + `_similar` + `_sure` | P: there is a third rule (a digit dropped or added), and the auto-apply (`sure`) needs extra conditions (`SURE_MM`, confidence, the number not already clean). "No other candidate" is the `sure` test, not the matching test |
| Twins: same cardstock, perimeters within 25 % | `_pair`: `0.25 * size[a]` | V |
| Cyclic alignment of reversed length sequences, NW style, gain 2 / 1 | `_align`, `_sequence`: gain 2 (≤ 1 mm), 1 (≤ 3 mm), **−∞ beyond 3 mm, free gaps** | P: it is a gap-free-cost LCS variant with forbidden mismatches, closer to weighted LCS than NW (NW has gap penalties). Also needs ≥ 3 numbered lines |
| Accept twin if share ≥ 0.7; greedy by share | `MIN_SHARE = 0.7`; share = agree / max(len) | V |
| Similarity with reflection on matched midpoints | `_similar` (SVD, no det fix, so reflection allowed); rejected if RMS > 1.5 mm | V (the RMS gate is unstated) |
| Flag line pairs > 1 mm | `Twin.off`, `mirror-mismatches.csv` | V |
| Missing fold added "in the style printed under it" | `mirror_fixes`: added only if `detector.fold_kind` finds fold evidence on the scan, **with the twin's style** (`gap["kind"]`); mismatched styles are reconciled to the clearer scan | P: the style comes from the twin; the scan only gates it. Style reconciliation is not described |
| Corners 0.2–1.5 mm apart move to the mirrored mean | `corner_moves`: mutual nearest, refit on corners, `MOVE_MIN/MAX_MM` | V |
| (not in paper) a twin that "folded cleanly in the last 3D fold" is taken as truth, and corners copy up to 3 mm | `corner_moves(clean)`, `COPY_MAX_MM = 3.0`, `fold3d-inspect.csv` | **N in paper.** This is a 3D → 2D feedback loop. Figure 1 shows none of it |
| (not in paper) corners nudged from the 3D fold | `__main__.py`: applies `fold3d-moves.csv` (461 rows exist in `bowser-fresh/`) | **N in paper.** Same loop. It matters for the "unsupervised" claim; see §3 |
| Glue predictions from twins | `Symmetry.predictions` (needs a clean twin pair and a partner with its own twin) | V |
| Self-checks: missed corner, clipped corner, piece traced twice, scale error | `checks.py` (junctions, `ink_past`, contours, scale with `SCALE_MM = 0.3`, ≥ 5 pairs) | V |
| Fold-up glues each pair and measures the gap | `fold3d.py`: rigid faces, robust reweighting, glues that gape > **3 mm** are let go (`gape_mm = 3.0`); hand pins weigh 4× and are never let go; twin corners tied in 3D (weight 0.5) | P: the closing threshold (3 mm), the let-go rule, the pins and the mirror tie are all unstated, and each one changes what "closes" means |
| Answers saved as CSV and replayed; corrections teach the matcher | `corrections.py`, `__main__.py`; also `confirmed-*.csv` files are merged in | V |

### Section 3.4 (output): `svg.py`, `export.py`

| Claim | Code | Verdict |
|---|---|---|
| One layer per cardstock, one group per piece with Cut / Score / Draw; hidden Review layer | `svg.py:105-150` | V (there is also a "Source" layer holding the scan at 35 % opacity. Worth a word, since it is what the user deletes before upload) |
| Export to inches and short labels | `export.py` | V |

---

## 2. Numbers, table and figures vs data

"right" and "wrong" below are as defined in `scripts/compare-runs.py`: each number of a run is
matched to the nearest number of the corrected build on the same piece (< 4 mm). It is **right** if
its status is ok and its partner *piece* equals the corrected build's partner piece.

| Paper claim | Evidence | Verdict |
|---|---|---|
| 24 US Letter pages, 21 with pieces | `pdfinfo bowser.pdf`: 24 pages, 612×792 pt; `report/data/bowser/pages.csv`: pages 4–24 | V |
| six cardstock colours | `pages.csv`: Red, Orange, Green, Yellow, Gold, White | V for page labels. The YAML (and so Figure 3's colours) uses **7** (adds brown), so say "six printed colours" |
| 178 pieces | `numbers.csv` `pieces,178`; both `edge-numbers.csv` files have 178 pieces | V |
| 1569 numbers (all answers) / 1575 (no answers) | `numbers.csv`; `bowser*/edge-numbers.csv` row counts | V. The abstract says "1569 edge numbers" but 67 % is 1059 / **1575**. Say once that the counts differ because the fresh run reads 6 extra marks (10 have no match in the corrected build) |
| trusted reads 970 / 1487 | `ocr_trusted` yes counts | V. But in the corrected build, hand-answered marks are trusted by definition (`read_all`: `known` gets `confident = True`), so 1487 is not a reader score. Relabel or drop the row |
| right 1059 / 1548, wrong 10 / 0, open 496 / 21 | `comparison/runs.csv` | V. 1548 vs itself is right by construction (see §3) |
| mirror twin pairs 86 / 89 | twin pairs in `mirror-mismatches.csv` | V. All 86 fresh pairs are among the 89. Note that 89 × 2 = 178, so **every** piece has a twin here ("most pieces" undersells it) |
| 3D: glues closing 527 / 631 | `bowser-fresh/fold3d-glue.csv`: 494 "closes" + 33 "proposed, closes"; 631 rows include 40 "proposed by shape only, not glued", 29 proposed not closing, 20 let go, 15 not closing | P: 33 of the 527 are pairs **proposed by shape**, not read numbers. Say so, or report 494 / 529 glues from read numbers (the other 102 rows are shape proposals) |
| 3D connected parts 3 / 1 | `comparison.md` §1; `fold3d-baseline.csv` parts 1 | V |
| (omitted) openings 44 / 20, expected 4 | `comparison.md` §1, `bowser/fold3d-baseline.csv`: openings 20 vs reference 4 | **Omitted, and it matters**: the "closed surface" has 16 unexpected holes |
| all 754 glued edges close, median gap 0.1 mm | `bowser/fold3d-glue.csv`: 754 rows, all "closes", median 0.1, **max 2.4 mm, 11 over 1 mm**; `fold3d.json` lists 755 glues | P: true at the solver's 3 mm tolerance. But there are 774 clean pairs (1548 / 2); 22 of them are absent from `fold3d-glue.csv`, 4 of those 22 are cross-cardstock joins in `fold3d-joins.csv`, and **4 of the 151 cross-cardstock joins do not close** there. `fold3d.json` also lists 9 wrong-way folds. "All 754 close" needs "within-cardstock, within 3 mm" or a reconciled count |
| human input 1079 = 670 + 90 + 319; 14 changed | `numbers.csv` (cards 670 / 14, lines 90, piece_edits 319) | V as arithmetic. **Undercounts** the human input (§3) |
| 6 of 21 open are eye-opening edges that glue to nothing | `edge-numbers.csv`: 6 "unpaired" (403 tab, 784, 789, 790, 792, 904) | P: the count matches, but nothing in the data says these are eye-opening edges. One is a **tab** (403), and a tab must glue to something. Check it |
| outlines median 0.60 mm, 90 % within 1.4 mm, 3 pieces > 3 mm | `comparison/shapes.csv` (`compare-shapes.py`) | V (p90 = 1.30 mm, so "within 1.4" holds). The metric is the **per-piece Hausdorff distance to the corrected build**, not "lie a median 0.6 mm from". Name it |
| 262 corner moves, 10 added folds | `bowser-fresh/mirror-fixes.csv`: 262 move, 10 add | V. Their correctness was not measured |
| 14 changed answers; reader without them reads 299 differently | `comparison.md` §2 | V (text only; no committed CSV backs the 299) |
| answers in rounds: 1059, 1154, 1252, 1326, 1400 | `runs.csv` loop1–4 | V |
| "roughly 550 answers would rebuild the checked result" | extrapolated | **Overclaim.** Gains per 100 are 95, 98, 74, 74, so the 160 still open need ≈ 220 more, which gives ≈ 620. The oracle only answers *numbers*. And "rebuild" ignores the 3D inputs, and that line and piece edits were needed to reach 1550 (`hand3`) |
| "half the 1079 rows" | | P: answers and rows are different units (178 of the 319 piece edits are "ok" clicks) |
| "an ablation ... a third of the fixes wrong (38 of 112)" | `comparison.md` §4: 74 right, 25 original was right, 13 neither | V numbers. **Not an ablation**: it is an error analysis of the automatic fixes. Rename |
| Figure 1 (pipeline) | | P: it is missing the 3D → trace feedback (corner nudges, clean-twin copy) |
| Figure 2 (schematic) | ours, not from the template | V. Fine |
| Figure 3 (rendered mask), "all 754 close" | | see the 754 row |

---

## 3. Unsupported claims, missing baselines, unclear evaluation

**E1. The reference is not ground truth, and it is circular.** "Right partner", "0.6 mm" and the
rebuild are scored against the corrected build, which is this pipeline plus hand edits. Errors that
both builds share (scale, systematic offsets, a pair both builds get wrong) are invisible, and the
corrected build scores 1548 / 1548 right against itself. Fix: call it "agreement with the
hand-checked build" throughout, and give one independent check. Options: the 3D closure, a few
pieces measured by hand on the printed sheet, or the synthetic benchmark.

**E2. "Unsupervised" needs a definition.** The fresh run keeps the cardstock per piece from the YAML
(stated in `comparison.md`, not in the paper). It also may apply `fold3d-moves.csv` and
`fold3d-inspect.csv` from an earlier 3D fold of the same run, both present in `bowser-fresh/`. State
how many trace ↔ 3D rounds the fresh result had, and which files were present when the shapes were
measured.

**E3. Human input is undercounted.** Beyond the three CSVs, the corrected build used:
`confirmed-number-corrections.csv` (23 rows), `confirmed-piece-edits.csv` (6), `confirmed-number-lines.csv` (4),
17 `confirmed_joins` in the YAML, 20 hand-pinned glues (`fold3d-glue-pins.json`), tab overrides (8),
accepted corners, `fold3d-753-resolution.json`, `fold3d-upper-teeth-adjustment.json` and
`intentionally-removed-tabs.json`. `comparison.md` says itself: "plus the 3D sessions". Either count
them, or say the table counts only 2D answers and the 3D closure also needed N pins, joins and
overrides.

**E4. "All glued edges close" is a solver statement.** Pinned glues weigh 4× and are never let go,
twins are tied in 3D, and "closes" means ≤ 3 mm. Report the tolerance, the max gap (2.4 mm), the
openings (20 vs 4), the wrong-way folds (9) and the cross-cardstock joins (147 / 151), or reconcile them.

**E5. No baseline in the paper.** The intro argues that generic vectorizers fail on fabrication
meaning, but the paper shows no comparison. `bench/summary/summary.md` already scores A (contours +
Douglas–Peucker), B (Potrace) and C (centerline) on 12 synthetic templates with exact truth, but not
this pipeline. One small table (fold-as-cut fragments, F3 macro-F1, closed outlines) with "ours"
added would answer the obvious reviewer question. If there is no room, cite the benchmark as
supplementary.

**E6. One template, one designer.** This is stated, which is good. Still, every threshold (4.3 mm,
110 grey level, 0.6, 25 %, dash runs at 180 dpi) was tuned on this template. Say that the thresholds
were set on this template, and that there is no held-out page.

**E7. "Self-checking" / "checks itself".** The checks find inconsistency, not error. A consistent
misread (both copies read as the same wrong number, or a swap between two pairs that join the same
two pieces) passes the pairing check and counts as "right" in `compare-runs.py`. Soften "reliably"
in the last sentence.

**E8. Constraint wording.** "Every number is printed exactly twice ... one of which is a tab" is
contradicted by the 6 numbered edges that "glue to nothing". Say "almost every".

**E9. "Two independent readers".** They are not independent: the matcher is trained on Tesseract's
confident reads, so it inherits Tesseract's errors. Drop "independent" (abstract).

---

## 4. Double-blind, attribution and submission risks

| # | Issue | Evidence | Severity |
|---|---|---|---|
| D1 | The public repo README names the template ("Bowser papercraft mask"), the **same Sketchfab URL as [Pap17]**, and lives under the author's GitHub account. Searching the cited URL or "Bowser papercraft PDF SVG" leads to the author | `README.md:38` on main | High. Check EG's rule on public code; at least remove the Sketchfab URL and "Bowser" from the public README (or make the repo private) until decisions |
| D2 | Attribution mismatch: README credits **"Sweets The Cat"** (sweetsthecat.itch.io); the paper and YAML credit **"Papercraft Cat (Remy Del Toro)"** | `README.md:38` vs `eg2027.bib`, `bowser.yaml` `credit:` | High (the paper must cite the right designer). Verify which name is current and use one |
| D3 | Generative-AI policy. Per the project's notes on the call, only grammar-level generative-AI use is allowed. The draft text, figures and analysis were written by AI agents (every commit on the PR is by Claude) | `git log` of PR #98 | High. Read the call's exact wording and decide before submission. Not something I can judge for you |
| D4 | The paper draft is in a public repo, on PR branch #98 (and on main once merged), with the title | `paper/eg2027/` | Medium. Keep it out of main until after review, or move it to the private repo |
| D5 | PDF metadata | `pdfinfo`: Author "Paper ID xxx", Creator empty, no hostname | OK. Remember to fill in the ID |
| D6 | Source files contain the de-anonymised author name and repo URL (`authors.tex`, `main.tex` `\hypersetup`) | | Low (only matters if you upload the source) |
| D7 | `\notestrue` is still on (no `\todo` shows in the PDF now); the CCS TODO comment remains | `main.tex` | Low. Set `\notesfalse` before submission |
| D8 | Naming a Nintendo character in a figure caption | Figure 3 | Low. The trademark note is fine |

---

## 5. What a skeptical reviewer would ask

1. What is your ground truth? How do you know the 1548 "right" pairs are right?
2. How do you compare with Potrace, or contour + Douglas–Peucker, with a fold classifier on the
   same template? Why is there no baseline?
3. How many human-minutes did the "all answers" build take, including the 3D sessions?
4. "Closes" within what tolerance? Why are there 20 openings when 4 are expected?
5. Were the thresholds tuned on the test template? How would it do on a second designer or a
   different Pepakura line style?
6. What is the digit matcher? Why not fine-tune a CNN on the 1548 labels?
7. Is the mirror-twin alignment robust to near-symmetric but different pieces? (Answer available:
   86 / 86 fresh twins agree with the corrected 89.)
8. Is the 550-answer estimate measured or extrapolated?
9. Is the 3D fold-up a contribution? It is used as the strongest check but is described in two
   sentences.

---

## 6. Prioritised fixes

**Must fix (accuracy of claims)**

1. Reword every accuracy claim as agreement with the hand-checked build (abstract, Results, Table 1
   caption) and name the metrics: per-piece Hausdorff, and partner-piece agreement. [E1]
2. Qualify "all 754 glued edges close": give the 3 mm tolerance, the max 2.4 mm, within-cardstock
   only, and the 20 pins. Report or explain the openings (20 vs 4), the 9 wrong folds and the 4
   cross-cardstock joins that do not close, or rerun so they are 0 and the count reconciles with 774
   clean pairs. [E4, §2]
3. Count all human input, or label the row "2D answers" and add the 3D inputs. [E3]
4. Replace "roughly 550 answers would rebuild the checked result" with what was measured: 400 answers
   took the run to 1400 / 1575, with diminishing gains. [§2]
5. Rename "ablation" to "error analysis of the automatic fixes". [§2]
6. Fix the designer attribution (D2) and settle the generative-AI policy question (D3).

**Should fix (method fidelity)**

7. Algorithm 1: two rounds, not "until no change"; mention `_sharpen`; coverage counts dashes and digits.
8. "trained on the page itself" → "on the template itself"; add one clause on what the matcher is.
9. Describe the 3D → trace feedback (corner nudges, clean twin as truth) and add it to Figure 1, or
   state that the reported runs did not use it. [E2]
10. The NW alignment is really LCS-like with forbidden mismatches; it needs ≥ 3 lines and the fit has
    a 1.5 mm RMS gate. A missing fold takes the twin's style.
11. Table: mark the 33 shape-proposed glues in 527 / 631; relabel "trusted reads" for the answered build.
12. Check the claim that the 6 unpaired numbers are eye-opening edges (one is a tab, 403).

**Nice to have**

13. A baseline row from `bench/` (E5).
14. "most pieces have a twin" → "all 178 pieces have a twin here".
15. Drop "independent" for the readers (E9); soften "reliably" (E7); say "almost every number" (E8).
16. D1/D4: decide what stays public until notification (EG 2027 short papers: 22 Jan 2027 deadline).
