"""Shared helpers. All paths are relative to the project root unless absolute."""

import glob
from pathlib import Path

# Project root = the folder that holds these scripts, `weights/` and `items/`.
# (In the original project the scripts lived in `src/`, so this was `.parent.parent`.)
ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def _natural_key(p):
    # 2.png before 10.png, numbered files before named ones (e.g. test.jpeg)
    return (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.name)


def resolve_inputs(entries, exts=IMAGE_EXTS):
    """Expand a list of files / folders / glob patterns into a sorted file list.

    "04-bg"            every file with a matching extension in the folder
    "03-samples/1.jpeg"  a single file
    "04-bg/*.png"      a glob pattern
    """
    if isinstance(entries, (str, Path)):
        entries = [entries]
    found = []
    for entry in entries:
        p = Path(entry)
        p = p if p.is_absolute() else ROOT / p
        if p.is_dir():
            found += [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
        elif p.is_file():
            found.append(p)
        else:
            matches = [Path(m) for m in glob.glob(str(p))]
            if not matches:
                print(f"[MISSING] {entry}")
            found += [m for m in matches if m.is_file()]
    return sorted(dict.fromkeys(found), key=_natural_key)


def root_path(p):
    """Absolute path for a project-root-relative path (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
