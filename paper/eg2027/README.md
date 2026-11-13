# Eurographics 2027 short paper (issue #97)

Draft of a 4-page short paper on the PDF-to-SVG conversion (deadline 22 Jan 2027, 23:59 UTC).
Build with `make eg2027` from the repository root.

- `main.tex`: preamble copied from the official `EGauthorGuidelines-short.tex` (EG 2027 style
  kit, files in `egkit/`, found through `latexmkrc`). Current build: exactly 4 pages including
  references.
- `authors.tex`: `\anonymoustrue` for the double-blind submission, `\anonymousfalse` for the
  camera-ready version. The anonymous author line reads `Paper ID xxx`: put the SRM paper ID there. `\codelink` prints the repository only when de-anonymised.
- `abstract.tex`, `body.tex`: text. `\notesfalse` hides draft notes.
- `eg2027.bib`: references (check each against the publisher before submission).

No figure shows the template's pages or pieces: the template is free for personal use only.
Numbers come from `report/data/bowser/numbers.csv` and the no-hand-input rebuild (#93).
