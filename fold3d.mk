# Fold the traced pattern back into the 3D model (prototype, src/lowpoly_pdf_to_svg/fold3d.py).
#   make -f fold3d.mk          data/<project>/fold3d.obj (+ .mtl), fold3d.png preview,
#                              fold3d-glue.csv: every glued edge, its gap in the folded model and status,
#                              fold3d-baseline.csv: triangles and corners against the YAML `reference:`,
#                              fold3d-moves.csv: corners 0.2-1 mm off as `move` rows (piece-edits.csv columns)
#                              and fold3d-inspect.csv: pieces to look at by eye (corner >1 mm off, glue open)
#                              and fold3d.json for the viewer,
#                              fold3d-units.csv / fold3d-joins.csv: the modular glue plan (see `plan`)
#   make -f fold3d.mk plan     fold3d-units.csv and fold3d-joins.csv only, in seconds, no fold:
#                              pieces split by cardstock into glue units, and the glues between units
#   make -f fold3d.mk view     serve app/fold3d-viewer.html on localhost: the glued model beside the
#                              pattern, click a face in either to find it in the other (local only)
#   make -f fold3d.mk install  extra Python dependencies (scipy, shapely, matplotlib)
# Needs data/<project>/<project>.svg from a full `make` first.
# Checkpoints after piece placement and reuses the pose on retries or geometry edits;
# pass COLD=1 for a full rebuild.

PYTHON  ?= python3
CONFIG  ?= data/example/example.yaml
PROJECT ?= $(basename $(notdir $(CONFIG)))
OUT     ?= $(patsubst %/,%,$(dir $(CONFIG)))

PORT    ?= 8000
VIEW_ROOT ?= $(abspath $(OUT)/../..)
VIEWER_PATH ?= $(if $(wildcard $(VIEW_ROOT)/app/fold3d-viewer.html),app/fold3d-viewer.html,tools/lowpoly-pdf-to-svg/app/fold3d-viewer.html)

.PHONY: fold3d install plan view
fold3d: $(OUT)/fold3d.obj

$(OUT)/fold3d.obj: $(OUT)/$(PROJECT).svg src/lowpoly_pdf_to_svg/fold3d.py src/lowpoly_pdf_to_svg/glueplan.py
	$(PYTHON) -m src.lowpoly_pdf_to_svg.fold3d --config $(CONFIG) $(if $(filter 1,$(COLD)),--cold,)

plan: $(OUT)/$(PROJECT).svg
	$(PYTHON) -m src.lowpoly_pdf_to_svg.glueplan --config $(CONFIG)

install:
	$(PYTHON) -m pip install -r requirements-fold3d.txt

view:   # no rebuild: the fold takes about half an hour, run `make -f fold3d.mk` first
	@echo "Open http://127.0.0.1:$(PORT)/$(VIEWER_PATH)?json=/data/$(PROJECT)/fold3d.json&svg=/data/$(PROJECT)/$(PROJECT).svg"
	$(PYTHON) -m http.server $(PORT) --bind 127.0.0.1 --directory "$(VIEW_ROOT)"
