"""
Render a diagram from a structure JSON produced by build_structure.py.

The diagram is drawn horizontally in reading order (left_to_right /
top_to_bottom of the source image). Components stacked within one position are
drawn one above the other. Two output formats:
  image - item pictures from ITEMS_DIR placed on a background plate, plus a legend
  text  - ASCII boxes (handy for quick checks in the terminal)

Usage: edit the CONFIGURATION block below, then run `python src/render_diagram.py`.
"""

import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import IMAGE_EXTS, resolve_inputs, root_path

# ==============================================================================
# CONFIGURATION - paths are relative to the project root
# ==============================================================================
INPUTS = ["structures"]    # JSON files, folders or globs, e.g. ["structures/6.json", "structures/1*.json"]
OUTPUT_FORMAT = "image"    # "image", "text" or "both"
OUTPUT_DIR = "structures/diagrams"   # <image>.png / <image>.txt are written here; None = no files (text only)
PRINT_TO_TERMINAL = False  # print the ASCII diagram (text / both)

# --- image rendering ----------------------------------------------------------
# Where the item pictures come from:
#   "reference" - pre-cropped pictures in ITEMS_DIR (full / half chosen by rule)
#   "crop"      - rectangles cut from the real photo at each component's bbox, so
#                 sizes are the real ones (e.g. an "empty" spanning 3 slots)
#   "auto"      - crop when every component can be cropped, else reference
ITEM_SOURCE = "auto"
CROP_SOURCE_DIRS = ["03-samples"]   # original photos, looked up by the structure's image name
CROP_PADDING = 0           # extra pixels around each bbox
ITEMS_DIR = "items"        # <class>.png (full) and <class> - half.png (half)
BACKGROUND = "background.png"   # inside ITEMS_DIR; stretched to the diagram size
# Rotation (degrees, counter-clockwise) applied to an item picture before use.
# "female connector" was cropped from a vertical-axis photo (landscape); the
# diagram is horizontal, where slots are portrait.
ROTATE = {"female connector": 90}
CELL_HEIGHT = 240          # height of a column; a stack splits it between its items
PIN_HEIGHT = 70            # pins are drawn at this height, centred vertically
COLUMN_GAP = 24
PLATE_PADDING = (40, 30)   # horizontal, vertical padding between plate edge and items
CANVAS_COLOR = (255, 255, 255, 255)
LEGEND = True
# ==============================================================================

SLOT_W = 15      # inner width of a slot box
HEIGHT = 11      # total height of a column (box incl. borders)
GAP = 2          # spaces between columns
LABELS = {
    "female connector": ["Female", "connector"],
    "female connector mid": ["Female", "connector", "mid"],
    "female connector sm": ["Female", "connector", "small"],
    "male connector mid": ["Male", "connector", "mid"],
    "male connector sm": ["Male", "connector", "small"],
    "mixed": ["Mixed"],
    "empty": ["Empty"],
}
PIN_GLYPH = {"male pin": ["O"], "female pin": ["-", "|", "-"]}


def slot_box(cls, height):
    label = LABELS.get(cls, cls.split())
    inner = height - 2
    if len(label) > inner:                      # stacked boxes are short
        label = [" ".join(label)]
    top = (inner - len(label)) // 2
    rows = [" " * SLOT_W] * inner
    for i, text in enumerate(label):
        rows[top + i] = text[:SLOT_W].center(SLOT_W)
    return ["-" * (SLOT_W + 2)] + [f"|{r}|" for r in rows] + ["-" * (SLOT_W + 2)]


def pin_box(cls):
    glyph = PIN_GLYPH.get(cls, [cls[:1]])
    pad = (HEIGHT - len(glyph)) // 2
    rows = [""] * pad + glyph + [""] * (HEIGHT - pad - len(glyph))
    return [r.center(3) for r in rows]


def column(position):
    comps = position["components"]
    if len(comps) == 1 and comps[0]["kind"] == "pin":
        return pin_box(comps[0]["class"])
    n = len(comps)
    heights = [HEIGHT // n + (1 if i < HEIGHT % n else 0) for i in range(n)]
    rows = []
    for c, h in zip(comps, heights):
        rows += slot_box(c["class"], h) if c["kind"] == "slot" else pin_box(c["class"])[:h]
    return rows


def render(structure):
    cols = [column(p) for p in structure["sequence"]]
    if not cols:
        return "(no components detected)"
    height = max(len(c) for c in cols)
    body = []
    for r in range(height):
        body.append((" " * GAP).join(
            (c[r] if r < len(c) else "").ljust(len(c[0])) for c in cols))
    width = len(body[0]) + 4
    lines = ["-" * (width + 2), "|" + " " * width + "|"]
    lines += ["|  " + row + "  |" for row in body]
    lines += ["|" + " " * width + "|", "-" * (width + 2)]
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Image rendering
# ------------------------------------------------------------------------------
_cache = {}


def load_item(cls, half):
    """Item picture for a class (RGBA, rotated per ROTATE), or None if missing."""
    key = (cls, half)
    if key not in _cache:
        name = f"{cls} - half.png" if half else f"{cls}.png"
        path = root_path(ITEMS_DIR) / name
        img = None
        if path.exists():
            img = Image.open(path).convert("RGBA")
            if cls in ROTATE and not half:
                img = img.rotate(ROTATE[cls], expand=True)
        _cache[key] = img
    return _cache[key]


def font(size):
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit(img, max_w, max_h):
    scale = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)


def placeholder(cls, w, h):
    img = Image.new("RGBA", (w, h), (200, 60, 60, 255))
    ImageDraw.Draw(img).text((4, 4), cls, fill=(255, 255, 255, 255), font=font(11))
    return img


def use_half(position, comp):
    """Half picture for stacked items and for a lone half (flagged by build_structure)."""
    return len(position["components"]) > 1 or "stack_side" in comp


def column_images(position, turn=False):
    """List of (class, image) for a position, top to bottom, scaled to fit the cell."""
    comps = position["components"]
    if len(comps) == 1 and comps[0]["kind"] == "pin":
        c = comps[0]["class"]
        img = load_item(c, False) or placeholder(c, PIN_HEIGHT, PIN_HEIGHT)
        return [(c, fit(img, 10 ** 4, PIN_HEIGHT))]
    half_h = (CELL_HEIGHT - 8) // 2
    out = []
    for c in (reversed(comps) if turn else comps):
        cls = c["class"]
        half = use_half(position, c)
        h = half_h if half else CELL_HEIGHT
        img = load_item(cls, half) or load_item(cls, not half) or placeholder(cls, h, h)
        out.append((cls, fit(img, 10 ** 4, h)))
    return out


def legend_image(classes, width):
    """Rows of [thumbnail] class name, wrapped to the given width."""
    f, thumb_h, pad = font(15), 36, 12
    meas = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    entries = []
    for cls in classes:
        img = load_item(cls, False)
        thumb = fit(img, 10 ** 4, thumb_h) if img else placeholder(cls, thumb_h, thumb_h)
        tw = round(meas.textlength(cls, font=f))
        entries.append((cls, thumb, tw))
    rows, row, x = [], [], pad
    for e in entries:
        w = e[1].width + 8 + e[2] + 24
        if row and x + w > width - pad:
            rows.append(row)
            row, x = [], pad
        row.append(e)
        x += w
    rows.append(row)
    img = Image.new("RGBA", (width, pad + len(rows) * (thumb_h + 10)), CANVAS_COLOR)
    d = ImageDraw.Draw(img)
    y = pad // 2
    for row in rows:
        x = pad
        for cls, thumb, tw in row:
            img.alpha_composite(thumb, (x, y))
            d.text((x + thumb.width + 8, y + thumb_h // 2), cls, fill=(40, 40, 40, 255), font=f, anchor="lm")
            x += thumb.width + 8 + tw + 24
        y += thumb_h + 10
    return img


def find_source_photo(structure):
    stem = Path(structure["image"]).stem
    for d in CROP_SOURCE_DIRS:
        for ext in IMAGE_EXTS:
            p = root_path(d) / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def crop_columns(structure, photo_path=None):
    """Columns of (class, image) cut from the original photo, or None if impossible.

    Crops are rotated so the diagram always reads horizontally (vertical-axis
    photos are turned 90 degrees) and share one scale so real sizes are kept.
    """
    path = Path(photo_path) if photo_path else find_source_photo(structure)
    if path is None:
        return None, "source photo not found"
    photo = Image.open(path).convert("RGBA")
    size = structure["image_size"]
    if photo.size != (size["width"], size["height"]):
        return None, f"{path.name} is {photo.size}, structure was built on {size['width']}x{size['height']}"
    turn = structure["axis"] == "vertical"
    cols = []
    for p in structure["sequence"]:
        col = []
        # turned photos: larger image x ends up on top, so stack bottom-to-top
        for c in (reversed(p["components"]) if turn else p["components"]):
            x0, y0, x1, y1 = c["bbox"]
            pad = CROP_PADDING
            im = photo.crop((max(0, x0 - pad), max(0, y0 - pad),
                             min(photo.width, x1 + 1 + pad), min(photo.height, y1 + 1 + pad)))
            col.append((c["class"], im.rotate(90, expand=True) if turn else im))
        cols.append(col)
    tallest = max(sum(im.height for _, im in col) + 8 * (len(col) - 1) for col in cols)
    scale = CELL_HEIGHT / tallest
    return [[(cls, im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS))
             for cls, im in col] for col in cols], None


def render_image(structure, photo_path=None):
    positions = structure["sequence"]
    if not positions:
        return None
    cols = None
    if ITEM_SOURCE in ("crop", "auto"):
        cols, why = crop_columns(structure, photo_path)
        if cols is None:
            print(f"  {structure['image']}: crop not possible ({why}); using reference pictures")
    if cols is None:
        turn = structure["axis"] == "vertical"
        cols = [column_images(p, turn) for p in positions]
    col_w = [max(im.width for _, im in col) for col in cols]

    px, py = PLATE_PADDING
    plate_w = sum(col_w) + COLUMN_GAP * (len(cols) - 1) + 2 * px
    plate_h = CELL_HEIGHT + 2 * py
    plate = Image.open(root_path(ITEMS_DIR) / BACKGROUND).convert("RGBA").resize((plate_w, plate_h), Image.LANCZOS)

    x = px
    for col, w, position in zip(cols, col_w, positions):
        total = sum(im.height for _, im in col) + 8 * (len(col) - 1)
        # a lone half item sits on the side it was found, not in the middle
        side = position["components"][0].get("stack_side") if len(position["components"]) == 1 else None
        if side == "top":
            y = py
        elif side == "bottom":
            y = py + CELL_HEIGHT - total
        else:
            y = py + (CELL_HEIGHT - total) // 2
        for _, im in col:
            plate.alpha_composite(im, (x + (w - im.width) // 2, y))
            y += im.height + 8
        x += w + COLUMN_GAP

    classes = list(dict.fromkeys(c["class"] for p in positions for c in p["components"]))
    margin = 20
    parts = [plate] + ([legend_image(classes, plate_w)] if LEGEND else [])
    canvas = Image.new("RGBA", (plate_w + 2 * margin, sum(p.height for p in parts) + 2 * margin), CANVAS_COLOR)
    y = margin
    for p in parts:
        canvas.alpha_composite(p, (margin, y))
        y += p.height
    return canvas.convert("RGB")


def save_png(img, path, tries=5):
    """Save via a temp file; on Windows an overwrite can fail while an
    indexer/previewer briefly holds the old file, so retry."""
    tmp = path.with_name(path.stem + ".tmp.png")
    for attempt in range(tries):
        try:
            img.save(tmp)
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(0.3)


def main():
    out = root_path(OUTPUT_DIR) if OUTPUT_DIR else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    want_image = OUTPUT_FORMAT in ("image", "both")
    want_text = OUTPUT_FORMAT in ("text", "both")
    for f in resolve_inputs(INPUTS, exts={".json"}):
        s = json.loads(f.read_text(encoding="utf-8"))
        if want_text:
            text = render(s)
            if PRINT_TO_TERMINAL:
                print(f"\n== {s['image']}  ({s['reading_direction']}) ==")
                print(text)
            if out:
                (out / f"{f.stem}.txt").write_text(text + "\n", encoding="utf-8")
        if want_image:
            img = render_image(s)
            if img is None:
                print(f"{f.name}: no components, skipped")
            elif out:
                save_png(img, out / f"{f.stem}.png")
                print(f"{f.name} -> {out / (f.stem + '.png')}")
            else:
                print(f"{f.name}: set OUTPUT_DIR to save images")


if __name__ == "__main__":
    main()
