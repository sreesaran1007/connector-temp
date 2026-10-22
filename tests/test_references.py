"""Tests for references.py (no model needed: the model is replaced by a stub).

Run from 07-app:  .venv\\Scripts\\python -m unittest discover -s tests -t .
"""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from references import FULL_SIDE, THUMBNAIL_SIDE, ReferenceStore

POSITIONS = [("male pin",), ("female connector mid", "female pin"), ("female pin",)]


def make_photo(folder, name, size=(1200, 800), color=(90, 120, 200)):
    path = Path(folder) / name
    Image.new("RGB", size, color).save(path)
    return path


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)
        self.derived = []

    def derive(self, path):
        self.derived.append(path.name)
        return list(POSITIONS)

    def load(self, derive=None):
        store = ReferenceStore(self.folder)
        store.load(derive or self.derive)
        return store

    # -- reading the folder -------------------------------------------------
    def test_a_missing_folder_is_an_empty_list_not_a_crash(self):
        store = ReferenceStore(self.folder / "does-not-exist")
        store.load(self.derive)
        self.assertEqual(store.all(), [])

    def test_every_photo_becomes_a_connector_in_natural_order(self):
        for name in ("10.jpg", "2.png", "1.jpeg", "alpha.jpg"):
            make_photo(self.folder, name)
        store = self.load()
        self.assertEqual([r.id for r in store.all()], ["1", "2", "10", "alpha"])

    def test_the_shape_of_each_photo_is_recorded(self):
        make_photo(self.folder, "1.jpg", size=(300, 400))
        make_photo(self.folder, "2.jpg", size=(1600, 1200))
        ratios = {r.id: r.aspect_ratio for r in self.load().all()}
        self.assertEqual(ratios, {"1": 0.75, "2": 1.333})

    def test_a_phone_photo_with_exif_rotation_has_the_upright_shape(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # stored sideways, displayed upright
        Image.new("RGB", (400, 200), (1, 2, 3)).save(self.folder / "1.jpg", exif=exif)
        self.assertEqual(self.load().get("1").aspect_ratio, 0.5)

    def test_default_names_are_readable(self):
        for name in ("3.jpg", "left_side.jpg"):
            make_photo(self.folder, name)
        names = {r.id: r.name for r in self.load().all()}
        self.assertEqual(names, {"3": "Connector 3", "left_side": "Left Side"})

    def test_other_files_are_ignored(self):
        make_photo(self.folder, "1.jpg")
        (self.folder / "notes.txt").write_text("hi")
        (self.folder / "2.gif").write_bytes(b"GIF89a")
        self.assertEqual([r.id for r in self.load().all()], ["1"])

    def test_unsafe_names_are_skipped(self):
        make_photo(self.folder, "good.jpg")
        make_photo(self.folder, "has space.jpg")
        make_photo(self.folder, "dots.in.name.jpg")
        make_photo(self.folder, "-leading.jpg")
        self.assertEqual([r.id for r in self.load().all()], ["good"])

    def test_two_photos_with_one_id_keep_the_first(self):
        make_photo(self.folder, "1.jpg", color=(255, 0, 0))
        make_photo(self.folder, "1.png", color=(0, 255, 0))
        refs = self.load().all()
        self.assertEqual(len(refs), 1)
        self.assertEqual(self.derived, ["1.jpg"])

    # -- the JSON next to each photo ----------------------------------------
    def test_a_photo_without_json_is_read_by_the_model_and_saved(self):
        make_photo(self.folder, "1.jpg")
        self.load()
        saved = json.loads((self.folder / "1.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["name"], "Connector 1")
        self.assertEqual(saved["positions"], [list(p) for p in POSITIONS])
        self.assertIn("note", saved)

    def test_saved_json_is_used_next_time_without_asking_the_model(self):
        make_photo(self.folder, "1.jpg")
        self.load()
        self.derived.clear()
        store = self.load()
        self.assertEqual(self.derived, [], "the model was not asked again")
        self.assertEqual(store.get("1").positions, POSITIONS)

    def test_a_hand_edited_json_is_respected_and_never_overwritten(self):
        make_photo(self.folder, "1.jpg")
        edited = {"name": "Left panel", "positions": [["female pin"], ["male pin"]]}
        (self.folder / "1.json").write_text(json.dumps(edited), encoding="utf-8")
        store = self.load()
        self.assertEqual(self.derived, [])
        self.assertEqual(store.get("1").name, "Left panel")
        self.assertEqual(store.get("1").positions, [("female pin",), ("male pin",)])
        self.assertEqual(json.loads((self.folder / "1.json").read_text(encoding="utf-8")), edited)

    def test_stacks_in_the_json_are_sorted_so_their_order_does_not_matter(self):
        make_photo(self.folder, "1.jpg")
        (self.folder / "1.json").write_text(
            json.dumps({"positions": [["female pin", "female connector mid"]]}), encoding="utf-8"
        )
        self.assertEqual(self.load().get("1").positions, [("female connector mid", "female pin")])

    def test_a_broken_json_skips_only_that_connector(self):
        make_photo(self.folder, "1.jpg")
        make_photo(self.folder, "2.jpg")
        (self.folder / "1.json").write_text("{not json", encoding="utf-8")
        self.assertEqual([r.id for r in self.load().all()], ["2"])

    def test_malformed_positions_skip_that_connector(self):
        bad = [
            {"positions": []},
            {"positions": "oops"},
            {"positions": [[]]},
            {"positions": [["ok", 5]]},
            {"positions": [["ok"], "x"]},
            ["not", "an", "object"],
        ]
        for i, data in enumerate(bad, start=1):
            make_photo(self.folder, f"{i}.jpg")
            (self.folder / f"{i}.json").write_text(json.dumps(data), encoding="utf-8")
        make_photo(self.folder, "9.jpg")
        self.assertEqual([r.id for r in self.load().all()], ["9"])

    def test_a_photo_where_the_model_sees_nothing_is_skipped_and_not_saved(self):
        make_photo(self.folder, "1.jpg")
        store = self.load(derive=lambda path: [])
        self.assertEqual(store.all(), [])
        self.assertFalse((self.folder / "1.json").exists())

    def test_a_failing_model_skips_only_that_photo(self):
        make_photo(self.folder, "1.jpg")
        make_photo(self.folder, "2.jpg")

        def flaky(path):
            if path.stem == "1":
                raise RuntimeError("model crashed")
            return list(POSITIONS)

        self.assertEqual([r.id for r in self.load(derive=flaky).all()], ["2"])

    # -- lookups and images -------------------------------------------------
    def test_only_ids_from_the_folder_can_be_looked_up(self):
        make_photo(self.folder, "1.jpg")
        store = self.load()
        self.assertIsNotNone(store.get("1"))
        for bad in ("2", "../1", "..", "1.jpg", "", "C:\\x"):
            self.assertIsNone(store.get(bad), bad)
            self.assertIsNone(store.thumbnail(bad), bad)
            self.assertIsNone(store.full_image(bad), bad)

    def test_thumbnail_is_a_small_jpeg_and_is_cached(self):
        make_photo(self.folder, "1.jpg", size=(1600, 1200))
        store = self.load()
        data = store.thumbnail("1")
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(max(image.size), THUMBNAIL_SIDE)
        self.assertEqual(image.size, (480, 360), "aspect ratio kept")
        self.assertIs(store.thumbnail("1"), data, "second call reuses the first")

    def test_a_small_photo_is_not_enlarged(self):
        make_photo(self.folder, "1.jpg", size=(200, 100))
        image = Image.open(io.BytesIO(self.load().thumbnail("1")))
        self.assertEqual(image.size, (200, 100))

    def test_the_full_image_is_capped_in_size(self):
        make_photo(self.folder, "1.jpg", size=(3000, 1500))
        image = Image.open(io.BytesIO(self.load().full_image("1")))
        self.assertEqual(max(image.size), FULL_SIDE)

    def test_a_phone_photo_with_exif_rotation_is_returned_upright(self):
        path = self.folder / "1.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise to display
        Image.new("RGB", (400, 200), (10, 20, 30)).save(path, exif=exif)
        image = Image.open(io.BytesIO(self.load().thumbnail("1")))
        self.assertEqual(image.size, (200, 400))


OTHER_POSITIONS = [("female connector mid",), ("female connector sm",), ("male pin",)]


def shift_mtime(path, seconds):
    """Move a file's modified time, so a test does not depend on how finely the
    file system dates changes made a moment apart."""
    st = path.stat()
    ns = st.st_mtime_ns + int(seconds * 1_000_000_000)
    os.utime(path, ns=(ns, ns))


class LiveRefreshTests(unittest.TestCase):
    """Photos added, replaced, edited or removed while the server is running."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)
        self.derived = []
        self.result = list(POSITIONS)

    def derive(self, path):
        self.derived.append(path.name)
        if isinstance(self.result, Exception):
            raise self.result
        return list(self.result)

    def start(self):
        store = ReferenceStore(self.folder)
        store.load(self.derive)
        return store

    def replace_photo(self, name, size, color, seconds_later=2):
        path = make_photo(self.folder, name, size=size, color=color)
        shift_mtime(path, seconds_later)
        return path

    def saved(self, name="1.json"):
        return json.loads((self.folder / name).read_text(encoding="utf-8"))

    # -- the case that went wrong: replacing a photo --------------------------
    def test_a_replaced_photo_shows_its_new_shape_parts_and_picture(self):
        make_photo(self.folder, "1.jpg", size=(300, 400), color=(255, 0, 0))
        store = self.start()
        old = store.get("1")
        old_thumbnail = store.thumbnail("1")
        self.assertEqual(old.aspect_ratio, 0.75)

        self.result = list(OTHER_POSITIONS)
        self.replace_photo("1.jpg", size=(1200, 300), color=(0, 0, 255))
        store.refresh()

        new = store.get("1")
        self.assertEqual(new.aspect_ratio, 4.0)
        self.assertEqual(new.positions, sorted_positions(OTHER_POSITIONS))
        self.assertNotEqual(new.version, old.version)
        thumbnail = Image.open(io.BytesIO(store.thumbnail("1")))
        self.assertEqual(thumbnail.size, (480, 120), "the new picture, not the cached old one")
        self.assertNotEqual(store.thumbnail("1"), old_thumbnail)
        self.assertEqual(Image.open(io.BytesIO(store.full_image("1"))).size, (1200, 300))

    def test_the_connector_keeps_its_name_and_the_old_json_is_kept_aside(self):
        make_photo(self.folder, "1.jpg")
        store = self.start()
        data = self.saved()
        data["name"] = "Main connector"
        (self.folder / "1.json").write_text(json.dumps(data), encoding="utf-8")
        shift_mtime(self.folder / "1.json", 1)
        store.refresh()
        self.assertEqual(store.get("1").name, "Main connector")

        self.result = list(OTHER_POSITIONS)
        self.replace_photo("1.jpg", size=(900, 300), color=(1, 2, 3), seconds_later=5)
        store.refresh()

        self.assertEqual(store.get("1").name, "Main connector", "the name survives")
        self.assertEqual(self.saved()["positions"], [list(p) for p in OTHER_POSITIONS])
        self.assertEqual(self.saved()["name"], "Main connector")
        old = self.saved("1.json.old")
        self.assertEqual(old["name"], "Main connector")
        self.assertEqual(old["positions"], [list(p) for p in POSITIONS], "what was there before")

    def test_a_replacement_with_an_older_date_is_still_noticed(self):
        make_photo(self.folder, "1.jpg", color=(255, 0, 0))
        store = self.start()
        self.derived.clear()
        self.result = list(OTHER_POSITIONS)

        # A photo copied in from elsewhere can carry a "modified" date from last
        # week; it is recognised by its content instead.
        path = make_photo(self.folder, "1.jpg", size=(640, 480), color=(0, 255, 0))
        shift_mtime(path, -7 * 24 * 3600)
        store.refresh()

        self.assertEqual(self.derived, ["1.jpg"])
        self.assertEqual(store.get("1").positions, sorted_positions(OTHER_POSITIONS))

    def test_the_thumbnail_is_never_the_old_one_after_a_replacement(self):
        make_photo(self.folder, "1.jpg", size=(400, 300), color=(255, 0, 0))
        store = self.start()
        self.assertEqual(Image.open(io.BytesIO(store.thumbnail("1"))).size, (400, 300))
        self.replace_photo("1.jpg", size=(200, 100), color=(0, 0, 255))
        store.refresh()
        self.assertEqual(Image.open(io.BytesIO(store.thumbnail("1"))).size, (200, 100))

    def test_the_version_follows_the_content_not_the_date(self):
        path = make_photo(self.folder, "1.jpg", color=(255, 0, 0))
        first = self.start().get("1").version
        shift_mtime(path, 100)  # same bytes, different date
        self.assertEqual(self.start().get("1").version, first)
        make_photo(self.folder, "1.jpg", color=(0, 255, 0))
        self.assertNotEqual(self.start().get("1").version, first)

    # -- adding and removing --------------------------------------------------
    def test_a_new_photo_appears_without_a_restart(self):
        make_photo(self.folder, "1.jpg")
        store = self.start()
        make_photo(self.folder, "2.jpg")
        store.refresh()
        self.assertEqual([r.id for r in store.all()], ["1", "2"])
        self.assertEqual(self.derived, ["1.jpg", "2.jpg"])

    def test_a_removed_photo_disappears(self):
        make_photo(self.folder, "1.jpg")
        make_photo(self.folder, "2.jpg")
        store = self.start()
        (self.folder / "1.jpg").unlink()
        store.refresh()
        self.assertEqual([r.id for r in store.all()], ["2"])
        self.assertIsNone(store.get("1"))
        self.assertIsNone(store.thumbnail("1"))

    def test_the_folder_disappearing_is_an_empty_list_not_a_crash(self):
        make_photo(self.folder, "1.jpg")
        store = self.start()
        (self.folder / "1.jpg").unlink()
        (self.folder / "1.json").unlink()
        self.folder.rmdir()
        self.addCleanup(self.folder.mkdir, exist_ok=True)  # let the temp dir clean up
        store.refresh()
        self.assertEqual(store.all(), [])

    # -- doing no work when nothing changed -----------------------------------
    def test_refreshing_when_nothing_changed_does_no_work(self):
        make_photo(self.folder, "1.jpg")
        store = self.start()
        before = store.get("1")
        self.derived.clear()
        for _ in range(3):
            store.refresh()
        self.assertEqual(self.derived, [])
        self.assertIs(store.get("1"), before)

    def test_an_unchanged_connector_is_not_read_again_when_another_changes(self):
        make_photo(self.folder, "1.jpg")
        make_photo(self.folder, "2.jpg")
        store = self.start()
        one = store.get("1")
        self.derived.clear()
        self.replace_photo("2.jpg", size=(500, 500), color=(9, 9, 9))
        store.refresh()
        self.assertEqual(self.derived, ["2.jpg"])
        self.assertIs(store.get("1"), one)

    def test_refresh_needs_load_first(self):
        with self.assertRaises(RuntimeError):
            ReferenceStore(self.folder).refresh()

    # -- editing the JSON ------------------------------------------------------
    def test_a_hand_edit_is_picked_up_at_once_and_never_overwritten(self):
        make_photo(self.folder, "1.jpg")
        store = self.start()
        self.derived.clear()
        data = self.saved()
        data["positions"] = [["female pin"], ["male pin"]]
        text = json.dumps(data)
        (self.folder / "1.json").write_text(text, encoding="utf-8")
        shift_mtime(self.folder / "1.json", 3)

        store.refresh()

        self.assertEqual(store.get("1").positions, [("female pin",), ("male pin",)])
        self.assertEqual(self.derived, [], "the model was not asked")
        self.assertEqual((self.folder / "1.json").read_text(encoding="utf-8"), text)
        self.assertFalse((self.folder / "1.json.old").exists())

    def test_a_json_written_by_hand_is_respected_even_if_older_than_the_photo(self):
        make_photo(self.folder, "1.jpg")
        (self.folder / "1.json").write_text(
            json.dumps({"name": "Mine", "positions": [["male pin"]]}), encoding="utf-8"
        )
        shift_mtime(self.folder / "1.json", -3600)
        store = self.start()
        self.assertEqual(self.derived, [])
        self.assertEqual(store.get("1").name, "Mine")
        self.assertEqual(store.get("1").positions, [("male pin",)])

    def test_an_automatic_json_from_before_fingerprints_is_stale_only_if_older(self):
        legacy = {"name": "Old style", "note": "Derived automatically.", "positions": [["female pin"]]}
        # newer than the photo: kept
        make_photo(self.folder, "1.jpg")
        (self.folder / "1.json").write_text(json.dumps(legacy), encoding="utf-8")
        shift_mtime(self.folder / "1.json", 5)
        # older than the photo: stale
        make_photo(self.folder, "2.jpg")
        (self.folder / "2.json").write_text(json.dumps(legacy), encoding="utf-8")
        shift_mtime(self.folder / "2.json", -3600)

        store = self.start()

        self.assertEqual(store.get("1").positions, [("female pin",)])
        self.assertEqual(self.derived, ["2.jpg"])
        self.assertEqual(store.get("2").positions, sorted_positions(POSITIONS))
        self.assertEqual(store.get("2").name, "Old style")
        self.assertEqual(self.saved("2.json.old")["positions"], [["female pin"]])
        self.assertIn("photo_fingerprint", self.saved("2.json"), "now fingerprinted")

    def test_edits_made_before_a_replacement_are_kept_in_the_old_file(self):
        make_photo(self.folder, "1.jpg", color=(255, 0, 0))
        store = self.start()
        data = self.saved()
        data["positions"] = [["female pin"], ["male pin"]]
        (self.folder / "1.json").write_text(json.dumps(data), encoding="utf-8")
        shift_mtime(self.folder / "1.json", 1)
        store.refresh()
        self.assertEqual(store.get("1").positions, [("female pin",), ("male pin",)])

        self.replace_photo("1.jpg", size=(640, 480), color=(0, 0, 255), seconds_later=5)
        store.refresh()

        self.assertEqual(self.saved("1.json.old")["positions"], [["female pin"], ["male pin"]])
        self.assertEqual(store.get("1").positions, sorted_positions(POSITIONS), "read from the new photo")

    # -- things going wrong -----------------------------------------------------
    def test_if_the_model_fails_on_a_replacement_the_old_json_is_left_alone(self):
        make_photo(self.folder, "1.jpg", color=(255, 0, 0))
        store = self.start()
        before = (self.folder / "1.json").read_text(encoding="utf-8")

        self.result = RuntimeError("model crashed")
        self.replace_photo("1.jpg", size=(640, 480), color=(0, 0, 255))
        store.refresh()

        self.assertIsNone(store.get("1"), "not shown while it cannot be read")
        self.assertEqual((self.folder / "1.json").read_text(encoding="utf-8"), before)
        self.assertFalse((self.folder / "1.json.old").exists(), "nothing was replaced")

    def test_a_broken_photo_is_skipped_and_picked_up_once_fixed(self):
        (self.folder / "1.jpg").write_bytes(b"this is not an image")
        store = self.start()
        self.assertEqual(store.all(), [])

        make_photo(self.folder, "1.jpg")
        shift_mtime(self.folder / "1.jpg", 2)
        store.refresh()

        self.assertEqual([r.id for r in store.all()], ["1"])

    def test_two_photos_with_one_id_always_keep_the_same_one(self):
        make_photo(self.folder, "1.png", color=(0, 255, 0))
        make_photo(self.folder, "1.jpeg", color=(255, 0, 0))
        for _ in range(3):
            store = self.start()
            self.assertEqual(store.get("1").image_path.name, "1.jpeg")


def sorted_positions(positions):
    return [tuple(sorted(p)) for p in positions]


if __name__ == "__main__":
    unittest.main()
