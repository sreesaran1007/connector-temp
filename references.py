"""
The reference connectors the app lets the user choose from.

Every photo in the references folder (default: references/) is one connector.
To add or replace one, put its photo there; no restart is needed. The server
looks at the folder whenever the app asks for the list (the app's reload button
does that), so the change shows up there.

Each connector needs its list of parts ("positions"). The first time a photo is
seen the model reads it and the result is saved next to it as <name>.json:

    references/2.jpeg
    references/2.json     {"name": "Connector 2", "photo_fingerprint": "...",
                           "positions": [["male pin"], ...]}

The JSON is the source of truth for that photo. The model can misread a
reference photo, so open the JSON and correct it by hand if needed; it is not
overwritten while the photo stays the same. Edits are picked up as soon as the
file is saved.

If you REPLACE the photo, its JSON describes a picture that is gone, so it is
read again from the new photo. The old file is kept next to it as <name>.json.old
and the connector keeps its name. (The photo is recognised by its content, not
its date, so a replacement with an older "modified" date is noticed too.)

A JSON that you wrote yourself (without a "photo_fingerprint" or a "note") is
never replaced.
"""

import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger("references")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# The id comes from the file name and ends up in URLs, so keep it plain.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
THUMBNAIL_SIDE = 480
FULL_SIDE = 2048

_NOTE = (
    "Derived automatically from the photo by the model. Edit the positions if it "
    "got something wrong. If the photo is replaced, this file is read again from "
    "the new photo (the old one is kept as .json.old). Delete this file to derive "
    "it again."
)


@dataclass(frozen=True)
class Reference:
    id: str
    name: str
    image_path: Path
    positions: list  # list of sorted tuples of class names
    aspect_ratio: float  # width / height of the upright photo
    version: str  # changes whenever the photo changes; used to refresh caches


def _sort_key(path):
    # 2 before 10, numbered files before named ones; the extension breaks ties
    # so that "1.jpeg" and "1.png" are always ranked the same way.
    stem = path.stem
    return (not stem.isdigit(), int(stem) if stem.isdigit() else 0, stem.lower(), path.suffix.lower())


def _default_name(stem):
    return f"Connector {stem}" if stem.isdigit() else stem.replace("_", " ").replace("-", " ").title()


def _aspect_ratio(path):
    with Image.open(path) as source:
        width, height = ImageOps.exif_transpose(source).size
    return round(width / height, 3)


def _fingerprint(path):
    """A short identifier of a file's content."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _read_positions(data):
    """Validate the 'positions' of a reference JSON; raise ValueError if malformed."""
    positions = data.get("positions") if isinstance(data, dict) else None
    if not isinstance(positions, list) or not positions:
        raise ValueError("'positions' must be a non-empty list")
    out = []
    for position in positions:
        if not isinstance(position, list) or not position or not all(isinstance(c, str) and c for c in position):
            raise ValueError("each position must be a non-empty list of class names")
        out.append(tuple(sorted(position)))
    return out


def _is_stale(data, fingerprint, json_mtime_ns, image_mtime_ns):
    """Does this JSON describe a different photo than the one now in the folder?"""
    if not isinstance(data, dict):
        return False  # malformed: reported when the positions are read
    recorded = data.get("photo_fingerprint")
    if recorded is not None:
        return recorded != fingerprint
    if "note" in data:
        # Written automatically by an older version, before fingerprints
        # existed: the best clue left is the dates.
        return json_mtime_ns < image_mtime_ns
    return False  # written by hand: always respected


def _stat_sig(path):
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


class ReferenceStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self._items = {}
        self._sigs = {}
        self._folder_sig = None
        self._derive = None
        self._lock = threading.Lock()
        # JSON files this store saved during the current rebuild. Their dates change
        # because of us, not because the user did anything.
        self._written = set()
        # (id, version) -> JPEG bytes. A new version is a new key, so a replaced
        # photo can never be answered with its old thumbnail.
        self._thumbnails = {}
        self._images = {}

    def load(self, derive_positions):
        """Read the folder from scratch. `derive_positions(image_path)` returns the
        positions the model sees in a photo; it is only called for photos whose
        JSON is missing or out of date."""
        with self._lock:
            self._items = {}
            self._sigs = {}
            self._folder_sig = None
            self._thumbnails = {}
            self._images = {}
        self.refresh(derive_positions)

    def refresh(self, derive_positions=None):
        """Bring the connectors up to date with the folder. Cheap when nothing has
        changed (it only looks at file dates and sizes)."""
        with self._lock:
            if derive_positions is not None:
                self._derive = derive_positions
            if self._derive is None:
                raise RuntimeError("load() must be called before refresh()")
            entries = self._scan()
            folder_sig = tuple((name, sig) for name, _, sig in entries)
            if folder_sig == self._folder_sig:
                return
            self._written = set()
            self._rebuild()
            # Our own JSON saves changed the folder too; count those as seen. Any
            # other difference (the user touched something while we were reading)
            # is left to the next refresh.
            after = tuple((name, sig) for name, _, sig in self._scan())
            differing = {name for name, _ in set(after) ^ set(folder_sig)}
            self._folder_sig = after if differing <= self._written else folder_sig

    # -- looking at the folder ----------------------------------------------
    def _scan(self):
        """(file name, path, (mtime_ns, size)) for every photo and JSON file."""
        if not self.folder.is_dir():
            return []
        out = []
        with os.scandir(self.folder) as it:
            for entry in it:
                path = Path(entry.path)
                if not entry.is_file() or path.suffix.lower() not in IMAGE_EXTS | {".json"}:
                    continue
                try:
                    out.append((entry.name, path, _stat_sig(path)))
                except OSError:
                    continue  # vanished between listing and looking at it
        return sorted(out, key=lambda e: e[0])

    def _rebuild(self):
        if not self.folder.is_dir():
            log.warning("references folder not found: %s", self.folder)
            self._items, self._sigs = {}, {}
            self._thumbnails, self._images = {}, {}
            return

        images = sorted(
            (p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=_sort_key,
        )
        items, sigs = {}, {}
        for image in images:
            ref_id = image.stem
            if not _ID.match(ref_id):
                log.warning("skipping %s: use letters, digits, - or _ in the file name", image.name)
                continue
            if ref_id in items or ref_id in sigs:
                log.warning("skipping %s: another photo already uses the id %r", image.name, ref_id)
                continue
            sidecar = image.with_suffix(".json")
            try:
                sig = (_stat_sig(image), _stat_sig(sidecar) if sidecar.exists() else None)
            except OSError:
                continue
            sigs[ref_id] = sig
            old = self._items.get(ref_id)
            if old is not None and self._sigs.get(ref_id) == sig and old.image_path == image:
                items[ref_id] = old  # nothing changed: keep what we have
                continue
            try:
                items[ref_id] = self._load_one(image, ref_id)
                if sidecar.name in self._written:
                    sigs[ref_id] = (sig[0], _stat_sig(sidecar))
            except Exception:
                # Left out of `items` (so not shown), but its date and size stay
                # in `sigs`: it is tried again when the folder next changes.
                log.exception("skipping %s: could not read it", image.name)
        self._items = items
        self._sigs = sigs
        live = {(r.id, r.version) for r in items.values()}
        self._thumbnails = {k: v for k, v in self._thumbnails.items() if k in live}
        self._images = {k: v for k, v in self._images.items() if k in live}
        log.info("%d reference connector(s) ready", len(items))

    def _load_one(self, image, ref_id):
        sidecar = image.with_suffix(".json")
        fingerprint = _fingerprint(image)
        version = fingerprint[:12]

        if sidecar.exists():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            st = sidecar.stat()
            if not _is_stale(data, fingerprint, st.st_mtime_ns, image.stat().st_mtime_ns):
                positions = _read_positions(data)
                name = str(data.get("name") or _default_name(ref_id))
                return Reference(ref_id, name, image, positions, _aspect_ratio(image), version)
            name = str(data.get("name") or _default_name(ref_id))
            log.info("%s was replaced: reading the new photo", image.name)
            replacing = True
        else:
            name = _default_name(ref_id)
            replacing = False

        positions = list(self._derive(image))
        if not positions:
            raise ValueError("the model found no connector parts in this photo")
        text = json.dumps(
            {
                "name": name,
                "note": _NOTE,
                "photo_fingerprint": fingerprint,
                "positions": [list(p) for p in positions],
            },
            indent=2,
        ) + "\n"
        try:
            if replacing:
                shutil.copyfile(sidecar, sidecar.with_name(sidecar.name + ".old"))
            sidecar.write_text(text, encoding="utf-8")
            self._written.add(sidecar.name)
        except OSError:
            log.warning("could not save %s; it will be derived again next time", sidecar.name)
        return Reference(ref_id, name, image, positions, _aspect_ratio(image), version)

    # -- lookups ------------------------------------------------------------
    def all(self):
        return list(self._items.values())

    def get(self, ref_id):
        """The reference with this id, or None. Only ids found in the folder are
        known, so a request can never reach any other file."""
        return self._items.get(ref_id)

    def thumbnail(self, ref_id):
        """JPEG bytes of a small version of the photo, or None for an unknown id."""
        return self._cached(self._thumbnails, ref_id, THUMBNAIL_SIDE, quality=80)

    def full_image(self, ref_id):
        """JPEG bytes of the photo (upright, at most FULL_SIDE px), or None."""
        return self._cached(self._images, ref_id, FULL_SIDE, quality=90)

    def _cached(self, cache, ref_id, side, quality):
        reference = self._items.get(ref_id)
        if reference is None:
            return None
        key = (ref_id, reference.version)
        if key not in cache:
            with Image.open(reference.image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((side, side), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=quality)
            cache[key] = buffer.getvalue()
        return cache[key]
