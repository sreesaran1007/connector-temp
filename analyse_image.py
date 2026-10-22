"""
Single-image pipeline: photo in -> connector diagram image out.

Runs the segmentation model, builds the structure and renders the diagram image
in one call. Import `analyse()` from other code, or edit IMAGE below and run
`python src/analyse_image.py`.

    from analyse_image import analyse
    diagram, structure = analyse("03-samples/6.jpeg")   # PIL image + dict
"""

import json
from pathlib import Path

from build_structure import DEFAULT_CONF, MODEL_PATH, build_structure
from common import root_path
from render_diagram import render_image, save_png

# ==============================================================================
# CONFIGURATION - paths are relative to the project root
# ==============================================================================
IMAGE = "03-samples/1.jpeg"          # the photo to analyse
OUTPUT_DIR = "structures/analysed"   # <image>.png (and .json) are written here; None = don't save
SAVE_STRUCTURE = True                # also write <image>.json next to the diagram
# Model and confidence default to the ones in build_structure.py
# ==============================================================================

_model = None


def get_model(model_path=MODEL_PATH):
    """Load the model once and reuse it between calls."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(root_path(model_path)))
    return _model


def analyse(image_path, conf=DEFAULT_CONF, output_dir=None, save_structure=False):
    """Analyse one photo. Returns (diagram PIL.Image or None, structure dict).

    The diagram is None when nothing was detected. If output_dir is given the
    diagram (and optionally the structure JSON) is also saved there.
    """
    path = root_path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    structure = build_structure(get_model(), path, conf)
    diagram = render_image(structure, photo_path=path)

    if output_dir:
        out = root_path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if diagram is not None:
            save_png(diagram, out / f"{path.stem}.png")
        if save_structure:
            (out / f"{path.stem}.json").write_text(json.dumps(structure, indent=2), encoding="utf-8")
    return diagram, structure


def main():
    diagram, structure = analyse(IMAGE, output_dir=OUTPUT_DIR, save_structure=SAVE_STRUCTURE)
    classes = [" / ".join(c["class"] for c in p["components"]) for p in structure["sequence"]]
    print(f"{Path(IMAGE).name}: {' | '.join(classes) or 'nothing detected'}")
    for w in structure["warnings"]:
        print(f"  warning: {w}")
    if diagram is not None and OUTPUT_DIR:
        print(f"Diagram saved to {root_path(OUTPUT_DIR) / (Path(IMAGE).stem + '.png')}")


if __name__ == "__main__":
    main()
