# Eurographics 2027 short paper (issue #97)

Draft of a 4-page short paper on the PDF-to-SVG conversion (deadline 22 Jan 2027, 23:59 UTC).
Build with `make eg2027` from the repository root.

- `main.tex`: preamble. Uses the official `egpubl.cls` when it is in this folder, otherwise a
  stand-in layout (A4, two columns, 9 pt Times). The official short-paper package is only
  available from the EG SRM system; copy its `.cls`, `.sty` and `.bst` files here and align the
  official branch of the preamble with the package's own template file.
- `authors.tex`: `\anonymoustrue` for the double-blind submission, `\anonymousfalse` for the
  camera-ready version. `\codelink` prints the repository only when de-anonymised.
- `abstract.tex`, `body.tex`: text. `\notesfalse` hides draft notes.
- `eg2027.bib`: references (check each against the publisher before submission).

No figure shows the template's pages or pieces: the template is free for personal use only.
Numbers come from `report/data/bowser/numbers.csv` and the no-hand-input rebuild (#93).
