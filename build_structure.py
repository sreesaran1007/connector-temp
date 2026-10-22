"""
Build a structured representation of a connector from YOLO segmentation results.

Pipeline: image -> YOLO-seg predictions -> de-duplicated components ->
ordered sequence along the connector's main axis -> JSON structure.

This module only produces the structure (no drawing). Use render_diagram.py to
visualise a structure JSON. The JSON is the unit intended for later comparison
against a user-created structure.

Usage: edit the CONFIGURATION block below, then run `python src/build_structure.py`.
"""

import json

import cv2
import numpy as np

from common import resolve_inputs, root_path

# ==============================================================================
# CONFIGURATION - paths are relative to the project root
# ==============================================================================
MODEL_PATH = "weights/best.pt"
INPUTS = ["03-samples"]             # files, folders or globs, e.g. ["04-bg/6.png", "04-bg/1*.png"]
OUTPUT_DIR = "structures"      # <image>.json is written here
DEFAULT_CONF = 0.25
# ==============================================================================

# Class groups. Names come from the model (dataset-seg/data.yaml); only the
# grouping below is a project decision.
SLOT_CLASSES = {
    "female connector", "female connector mid", "female connector sm",
    "male connector mid", "male connector sm", "mixed", "empty",
}
PIN_CLASSES = {"male pin", "female pin"}

# Two detections overlapping this much (intersection / smaller area) are the
# same physical object; keep the more confident one.
DUPLICATE_OVERLAP = 0.6
# A pin overlapping a slot this much is considered inside it, not in sequence.
CONTAINED_OVERLAP = 0.5
# Items share a position when their spans along the axis overlap this much
# (relative to the shorter span).
STACK_OVERLAP = 0.5
# A lone slot narrower across the axis than this fraction of the full slot
# height is one half of a double slot whose other half was not detected.
HALF_RATIO = 0.65
LOW_CONF = 0.5
# Axis is horizontal/vertical if the item centres spread this much more in one
# direction than in the other.
MIN_AXIS_RATIO = 1.5


def poly_mask(poly, shape):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def overlap_of_smaller(a, b):
    inter = np.logical_and(a, b).sum()
    smaller = min(a.sum(), b.sum())
    return float(inter / smaller) if smaller else 0.0


def detect(model, image_path, conf):
    """Run the model; return (image_shape, list of raw detections)."""
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    shape = result.orig_shape
    dets = []
    if result.masks is None:
        return shape, dets
    for box, poly in zip(result.boxes, result.masks.xy):
        if len(poly) < 3:
            continue
        cls_id = int(box.cls[0])
        dets.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 3),
            "polygon": poly,
        })
    return shape, dets


def dedupe(dets, shape):
    """Drop lower-confidence detections that overlap a kept one (any class)."""
    masks = [poly_mask(d["polygon"], shape) for d in dets]
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["confidence"])
    kept, dropped = [], []
    for i in order:
        dup_of = next((j for j in kept
                       if overlap_of_smaller(masks[i], masks[j]) >= DUPLICATE_OVERLAP), None)
        if dup_of is None:
            kept.append(i)
        else:
            dropped.append({"class_name": dets[i]["class_name"],
                            "confidence": dets[i]["confidence"],
                            "duplicate_of": dets[dup_of]["class_name"]})
    return [(dets[i], masks[i]) for i in kept], dropped


def describe(det, mask):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return {
        "class": det["class_name"],
        "kind": "slot" if det["class_name"] in SLOT_CLASSES else "pin",
        "confidence": det["confidence"],
        "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        "bbox": [x0, y0, x1, y1],
        "width": x1 - x0 + 1,
        "height": y1 - y0 + 1,
        "area": int(mask.sum()),
    }


def main_axis(items):
    """'horizontal' or 'vertical' from the spread of the item centres.

    Falls back to the elongation of the combined bounding box when the centres
    alone are not decisive (e.g. a single item).
    """
    pts = np.array([i["centroid"] for i in items])
    spread_x, spread_y = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    if spread_x > MIN_AXIS_RATIO * spread_y:
        return "horizontal"
    if spread_y > MIN_AXIS_RATIO * spread_x:
        return "vertical"
    boxes = np.array([i["bbox"] for i in items])
    w = boxes[:, 2].max() - boxes[:, 0].min()
    h = boxes[:, 3].max() - boxes[:, 1].min()
    return "horizontal" if w >= h else "vertical"


def resolve_end_pins(sequence):
    """Apply the rule: one side of the connector carries only one pin type.

    The pins at each end of the sequence form that side's pin group. If a group
    holds both male and female pins and the opposite side has a female pin, the
    female pins in the conflicting group are dropped (the model sometimes reads
    the screw holder beside a male pin as a female pin). If the opposite side
    has no female pin the conflict is left as is and reported.

    Returns (sequence, discarded_pins, warnings).
    """
    def run(items):
        out = []
        for it in items:
            if it["kind"] != "pin":
                break
            out.append(it)
        return out

    start, end = run(sequence), run(sequence[::-1])
    if len(start) == len(sequence):          # only pins: no sides to compare
        return sequence, [], []

    sides = {"start": start, "end": end}
    other = {"start": end, "end": start}
    drop, warnings = [], []
    for name, group in sides.items():
        classes = {p["class"] for p in group}
        if classes != {"male pin", "female pin"}:
            continue
        if any(p["class"] == "female pin" for p in other[name]):
            drop += [p for p in group if p["class"] == "female pin"]
        else:
            warnings.append(f"male and female pins on the same side ({name}) and no female pin "
                            "on the opposite side; left unresolved")
    rejected = [{"class": p["class"], "confidence": p["confidence"],
                 "reason": "male and female pin on the same side; kept male"} for p in drop]
    return [i for i in sequence if not any(i is d for d in drop)], rejected, warnings


def mark_half_slots(sequence, axis):
    """Flag a lone slot that is only half the height of a full slot.

    A double slot holds two half items stacked across the axis; when the model
    finds only one of them the position has a single, narrow component. Its
    `stack_side` ("top" / "bottom", in the horizontal diagram orientation)
    records which half was found, so the missing half is not mistaken for a
    centred item. Vertical-axis photos are turned 90 degrees counter-clockwise
    for the diagram, which flips the across direction (image x -> -y).
    """
    def diag_y(comp):
        return comp["centroid"][1] if axis == "horizontal" else -comp["centroid"][0]

    def diag_span(comp):
        x0, y0, x1, y1 = comp["bbox"]
        return (y0, y1) if axis == "horizontal" else (-x1, -x0)

    slots = [c for p in sequence for c in p["components"] if c["kind"] == "slot"]
    if not slots:
        return
    spans = [diag_span(c) for c in slots]
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    mid = (lo + hi) / 2
    full = max(
        max(diag_span(c)[1] for c in p["components"]) - min(diag_span(c)[0] for c in p["components"])
        for p in sequence if any(c["kind"] == "slot" for c in p["components"]))

    for p in sequence:
        comps = p["components"]
        if len(comps) != 1 or comps[0]["kind"] != "slot":
            continue
        c = comps[0]
        if c["length_across_axis"] < HALF_RATIO * full:
            c["stack_side"] = "top" if diag_y(c) < mid else "bottom"


def build_structure(model, image_path, conf=DEFAULT_CONF):
    shape, raw = detect(model, image_path, conf)
    kept, dropped = dedupe(raw, shape)
    items = [describe(d, m) for d, m in kept]
    masks = {id(it): m for it, (_, m) in zip(items, kept)}

    # Pins lying inside a slot are properties of that slot, not sequence items.
    slots = [i for i in items if i["kind"] == "slot"]
    sequence, contained = [], []
    for it in items:
        parent = None
        if it["kind"] == "pin":
            parent = next((s for s in slots
                           if overlap_of_smaller(masks[id(it)], masks[id(s)]) >= CONTAINED_OVERLAP), None)
        (contained if parent else sequence).append((it, parent))

    sequence = [it for it, _ in sequence]
    structure = {
        "image": image_path.name,
        "image_size": {"width": int(shape[1]), "height": int(shape[0])},
        "confidence_threshold": conf,
        "axis": None,
        "reading_direction": None,
        "sequence": [],
        "contained": [],
        "discarded_duplicates": dropped,
        "warnings": [],
    }
    if not sequence:
        structure["warnings"].append("no components detected")
        return structure

    axis = main_axis(sequence)
    k = 0 if axis == "horizontal" else 1          # coordinate index along axis
    sequence.sort(key=lambda i: i["centroid"][k])
    sequence, rejected, notes = resolve_end_pins(sequence)
    structure["discarded_pins"] = rejected
    structure["axis"] = axis
    structure["reading_direction"] = "left_to_right" if axis == "horizontal" else "top_to_bottom"

    # Size along / across the axis, relative to the median slot, for comparison
    # that does not depend on camera distance.
    along = (lambda i: i["width"]) if axis == "horizontal" else (lambda i: i["height"])
    across = (lambda i: i["height"]) if axis == "horizontal" else (lambda i: i["width"])
    ref_slots = [along(i) for i in sequence if i["kind"] == "slot"]
    ref = float(np.median(ref_slots)) if ref_slots else float(np.median([along(i) for i in sequence]))
    ref = max(ref, 1.0)

    # Group items that share the same span along the axis into one position
    # (components stacked across the axis, e.g. two rows of small connectors).
    groups = []
    for it in sequence:
        lo, hi = it["bbox"][k], it["bbox"][k + 2]
        g = groups[-1] if groups else None
        if g:
            shared = min(hi, g["hi"]) - max(lo, g["lo"])
            if shared >= STACK_OVERLAP * min(hi - lo, g["hi"] - g["lo"]):
                g["items"].append(it)
                g["lo"], g["hi"] = min(lo, g["lo"]), max(hi, g["hi"])
                continue
        groups.append({"lo": lo, "hi": hi, "items": [it]})

    item_pos = {}
    prev_hi = None
    for pos, g in enumerate(groups):
        g["items"].sort(key=lambda i: i["centroid"][1 - k])   # across-axis order
        comps = []
        for order, it in enumerate(g["items"]):
            item_pos[id(it)] = pos
            comps.append({
                "class": it["class"],
                "kind": it["kind"],
                "confidence": it["confidence"],
                "stack_index": order,
                "centroid": it["centroid"],
                "bbox": it["bbox"],
                "length_along_axis": along(it),
                "length_across_axis": across(it),
            })
        structure["sequence"].append({
            "position": pos,
            "relative_length": round((g["hi"] - g["lo"]) / ref, 2),
            "gap_to_previous": None if prev_hi is None else round((g["lo"] - prev_hi) / ref, 2),
            "components": comps,
        })
        prev_hi = g["hi"]

    mark_half_slots(structure["sequence"], axis)

    for pin, parent in contained:
        structure["contained"].append({
            "class": pin["class"], "confidence": pin["confidence"],
            "inside_position": item_pos[id(parent)],
        })

    # Sanity warnings
    structure["warnings"] += notes
    for e in structure["sequence"]:
        for c in e["components"]:
            if c["confidence"] < LOW_CONF:
                structure["warnings"].append(
                    f"position {e['position']} ({c['class']}) has low confidence {c['confidence']}")
        if len(e["components"]) > 1:
            structure["warnings"].append(
                f"position {e['position']} holds {len(e['components'])} stacked components")
        for c in e["components"]:
            if "stack_side" in c:
                structure["warnings"].append(
                    f"position {e['position']}: only the {c['stack_side']} half detected ({c['class']}); other half missing")
    return structure


def main():
    from ultralytics import YOLO
    model = YOLO(str(root_path(MODEL_PATH)))
    out = root_path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    for img in resolve_inputs(INPUTS):
        s = build_structure(model, img, DEFAULT_CONF)
        (out / f"{img.stem}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        seq = " | ".join("/".join(c["class"] for c in e["components"]) for e in s["sequence"]) or "-"
        print(f"{img.name:<10} {s['axis'] or '-':<10} {seq}")
        for w in s["warnings"]:
            print(f"    warning: {w}")


if __name__ == "__main__":
    main()
