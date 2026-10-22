"""End-to-end tests of server.py with the REAL model (takes about a minute: the
model loads once for the whole file).

The references are a temporary copy of tests/fixtures/ (frozen photos) with frozen
part lists, so these tests do not change when the photos in references/ are replaced.

Run from 07-app:  .venv\\Scripts\\python -m unittest discover -s tests -t .
"""

import base64
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

HERE = Path(__file__).resolve().parent.parent

# The server reads REFERENCES_DIR when it is imported, so set it first.
_TMP = tempfile.mkdtemp(prefix="server-test-refs-")
os.environ["REFERENCES_DIR"] = _TMP

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from tests.test_matching import P  # noqa: E402

FIXTURES = HERE / "tests" / "fixtures"

# What the model read from the four fixture photos (frozen: the tests check the
# server, not how well the model reads these particular pictures).
FIXTURE_LISTS = {
    "1": P("mp | F | Ms | Ms | Ms | fp"),
    "2": P("mp | Fm/Fm | Fs | Fs | Fs | Fm/F | fp"),
    "3": P("mp | Fm/F | Fs | Fm/F | Fs | Fs | fp"),
    "4": P("Fs | Mm | Ms | Ms | F | fp"),
}

_client_cm = None
client = None


def setUpModule():
    global _client_cm, client
    for n in ("1", "2", "3", "4"):
        shutil.copy(FIXTURES / f"{n}.jpeg", Path(_TMP) / f"{n}.jpeg")
        (Path(_TMP) / f"{n}.json").write_text(
            json.dumps({"name": f"Connector {n}", "positions": [list(p) for p in FIXTURE_LISTS[n]]}),
            encoding="utf-8",
        )
    _client_cm = TestClient(server.app)
    client = _client_cm.__enter__()  # starts the app: loads the model


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    shutil.rmtree(_TMP, ignore_errors=True)


def photo(n="2"):
    return (FIXTURES / f"{n}.jpeg").read_bytes()


def post_match(data, reference_id="2", filename="p.jpg", **extra):
    return client.post(
        "/match",
        files={"image": (filename, data, "image/jpeg")},
        data={"reference_id": reference_id, **extra},
    )


def jpeg_bytes(image, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", **kwargs)
    return buffer.getvalue()


class ReferenceEndpointTests(unittest.TestCase):
    def test_health(self):
        self.assertEqual(client.get("/health").json(), {"status": "ok"})

    def test_lists_the_four_connectors_in_order(self):
        body = client.get("/references").json()
        self.assertEqual([r["id"] for r in body], ["1", "2", "3", "4"])
        second = body[1]
        self.assertEqual(second["name"], "Connector 2")
        self.assertEqual(second["positions"], 7)
        for ref in body:
            with Image.open(FIXTURES / f"{ref['id']}.jpeg") as im:
                self.assertEqual(ref["aspect_ratio"], round(im.width / im.height, 3), ref["id"])
        self.assertGreater(body[0]["aspect_ratio"], 3, "the fixtures are wide strips")
        # The address carries the photo's version, so a replaced photo is fetched
        # again instead of being served from a browser's cache.
        self.assertRegex(second["version"], r"^[0-9a-f]{12}$")
        self.assertEqual(second["thumbnail_url"], f"/references/2/thumbnail?v={second['version']}")
        self.assertEqual(second["image_url"], f"/references/2/image?v={second['version']}")

    def test_thumbnail_is_a_small_jpeg(self):
        response = client.get("/references/1/thumbnail")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        image = Image.open(io.BytesIO(response.content))
        self.assertLessEqual(max(image.size), 480)

    def test_the_larger_image_is_a_jpeg(self):
        response = client.get("/references/3/image")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(response.content)).format, "JPEG")

    def test_unknown_or_hostile_ids_are_404(self):
        for bad in ("9", "..", "%2e%2e", "2.jpeg", "..%2f..%2fserver.py", "1%00"):
            for path in (f"/references/{bad}/thumbnail", f"/references/{bad}/image"):
                self.assertEqual(client.get(path).status_code, 404, path)


class LiveReferenceTests(unittest.TestCase):
    """The folder is looked at whenever the app asks, so a photo can be added,
    replaced or removed while the server keeps running (with the real model)."""

    def setUp(self):
        self.addCleanup(self._remove_connector_nine)

    @staticmethod
    def _remove_connector_nine():
        for name in ("9.jpeg", "9.json", "9.json.old"):
            (Path(_TMP) / name).unlink(missing_ok=True)
        client.get("/references")  # let the server notice, so other tests see four

    @staticmethod
    def entry(connector_id="9"):
        for item in client.get("/references").json():
            if item["id"] == connector_id:
                return item
        return None

    @staticmethod
    def put_photo(image):
        path = Path(_TMP) / "9.jpeg"
        image.save(path, "JPEG", quality=95)
        # A moment later than before, however finely the disk dates writes.
        later = path.stat().st_mtime_ns + 3_000_000_000
        os.utime(path, ns=(later, later))

    @staticmethod
    def widened(path):
        """The same connector on a canvas twice as wide: a different shape."""
        with Image.open(path) as src:
            src = src.convert("RGB")
            canvas = Image.new("RGB", (src.width * 2, src.height), (128, 128, 128))
            canvas.paste(src, (src.width // 2, 0))
        return canvas

    def test_a_photo_added_while_running_shows_up_with_its_picture(self):
        self.assertIsNone(self.entry())
        shutil.copy(FIXTURES / "1.jpeg", Path(_TMP) / "9.jpeg")

        item = self.entry()

        self.assertIsNotNone(item, "no restart needed")
        self.assertEqual(item["name"], "Connector 9")
        self.assertGreater(item["positions"], 0, "read by the model")
        self.assertTrue((Path(_TMP) / "9.json").exists(), "and saved next to the photo")
        picture = client.get(item["thumbnail_url"])
        self.assertEqual(picture.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(picture.content)).format, "JPEG")

    def test_a_replaced_photo_changes_shape_picture_and_address_at_once(self):
        shutil.copy(FIXTURES / "2.jpeg", Path(_TMP) / "9.jpeg")
        before = self.entry()
        picture_before = client.get(before["thumbnail_url"]).content
        size_before = Image.open(io.BytesIO(picture_before)).size

        self.put_photo(self.widened(FIXTURES / "2.jpeg"))
        after = self.entry()

        self.assertIsNotNone(after, "still listed: the model could read the new photo")
        self.assertGreater(after["aspect_ratio"], before["aspect_ratio"] * 1.5, "the new shape")
        self.assertNotEqual(after["version"], before["version"])
        self.assertNotEqual(after["thumbnail_url"], before["thumbnail_url"], "so a browser fetches it again")
        picture_after = client.get(after["thumbnail_url"]).content
        self.assertNotEqual(picture_after, picture_before, "the new picture, not the cached old one")
        self.assertGreater(Image.open(io.BytesIO(picture_after)).size[0] / Image.open(io.BytesIO(picture_after)).size[1],
                           size_before[0] / size_before[1])

    def test_a_replaced_photo_is_read_again_and_the_old_list_is_kept_aside(self):
        shutil.copy(FIXTURES / "2.jpeg", Path(_TMP) / "9.jpeg")
        self.entry()
        first = json.loads((Path(_TMP) / "9.json").read_text(encoding="utf-8"))

        self.put_photo(self.widened(FIXTURES / "2.jpeg"))
        self.entry()

        second = json.loads((Path(_TMP) / "9.json").read_text(encoding="utf-8"))
        self.assertNotEqual(second["photo_fingerprint"], first["photo_fingerprint"])
        self.assertEqual(second["name"], first["name"])
        kept = json.loads((Path(_TMP) / "9.json.old").read_text(encoding="utf-8"))
        self.assertEqual(kept["photo_fingerprint"], first["photo_fingerprint"])

    def test_a_removed_photo_disappears(self):
        shutil.copy(FIXTURES / "1.jpeg", Path(_TMP) / "9.jpeg")
        item = self.entry()
        self.assertIsNotNone(item)

        (Path(_TMP) / "9.jpeg").unlink()

        self.assertIsNone(self.entry())
        self.assertEqual(client.get("/references/9/thumbnail").status_code, 404)

    def test_the_connector_that_was_not_touched_is_unchanged(self):
        before = self.entry("3")
        shutil.copy(FIXTURES / "1.jpeg", Path(_TMP) / "9.jpeg")
        self.entry()
        self.assertEqual(self.entry("3"), before)

    def test_matching_uses_the_new_parts_of_a_replaced_photo(self):
        shutil.copy(FIXTURES / "2.jpeg", Path(_TMP) / "9.jpeg")
        self.entry()
        self.put_photo(self.widened(FIXTURES / "2.jpeg"))  # same connector, new picture

        body = post_match(photo("2"), "9").json()

        self.assertEqual(body["match"]["reference_id"], "9")
        self.assertTrue(body["match"]["matches"], "the new list was read from the new photo")


class MatchEndpointTests(unittest.TestCase):
    def test_a_photo_of_the_chosen_connector_matches(self):
        response = post_match(photo("2"), "2")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["detected"])
        match = body["match"]
        self.assertEqual(match["reference_id"], "2")
        self.assertEqual(match["reference_name"], "Connector 2")
        self.assertTrue(match["matches"])
        self.assertGreaterEqual(match["similarity"], 0.9)
        self.assertEqual(match["threshold"], 0.7)
        self.assertEqual(match["differences"], [])
        self.assertTrue(base64.b64decode(body["diagram_png_base64"]).startswith(b"\x89PNG"))

    def test_the_same_photo_against_another_connector_does_not_match(self):
        body = post_match(photo("2"), "3").json()
        self.assertFalse(body["match"]["matches"])
        self.assertLess(body["match"]["similarity"], 0.7)
        self.assertTrue(body["match"]["differences"], "says what is different")
        self.assertTrue(body["detected"], "the parts are still shown")

    def test_every_reference_photo_matches_only_its_own_connector(self):
        for photo_of in ("1", "2", "3", "4"):
            data = photo(photo_of)
            for chosen in ("1", "2", "3", "4"):
                match = post_match(data, chosen).json()["match"]
                self.assertEqual(match["matches"], photo_of == chosen, f"photo {photo_of} vs {chosen}")

    def test_a_phone_style_rotated_photo_still_matches(self):
        original = Image.open(FIXTURES / "2.jpeg").convert("RGB")
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise to display
        stored = original.rotate(90, expand=True)
        body = post_match(jpeg_bytes(stored, quality=95, exif=exif), "2").json()
        self.assertTrue(body["match"]["matches"], body["match"])

    def test_a_photo_with_no_connector_in_it_does_not_match(self):
        blank = jpeg_bytes(Image.new("RGB", (800, 600), (40, 40, 40)))
        body = post_match(blank, "2").json()
        self.assertFalse(body["detected"])
        self.assertIsNone(body["diagram_png_base64"])
        self.assertFalse(body["match"]["matches"])
        self.assertEqual(body["match"]["similarity"], 0.0)

    def test_an_unknown_connector_is_refused_before_the_photo_is_read(self):
        response = post_match(b"this is not even an image", "99")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not known", response.json()["detail"])

    def test_a_hostile_connector_id_is_refused(self):
        for bad in ("../2", "..", "2.jpeg", "", "2 "):
            self.assertIn(post_match(photo("2"), bad).status_code, (404, 422), repr(bad))

    def test_a_missing_connector_id_is_a_validation_error(self):
        response = client.post("/match", files={"image": ("p.jpg", photo("2"), "image/jpeg")})
        self.assertEqual(response.status_code, 422)

    def test_a_bad_upload_is_explained(self):
        self.assertEqual(post_match(b"not an image", "2").status_code, 400)
        self.assertEqual(post_match(b"", "2").status_code, 400)
        gif = io.BytesIO()
        Image.new("RGB", (20, 20)).save(gif, "GIF")
        self.assertEqual(post_match(gif.getvalue(), "2").status_code, 415)
        self.assertEqual(post_match(b"\xff" * (16 * 1024 * 1024), "2").status_code, 413)

    def test_the_threshold_can_be_changed(self):
        with mock.patch.object(server, "MATCH_THRESHOLD", 1.01):
            body = post_match(photo("2"), "2").json()
        self.assertFalse(body["match"]["matches"], "nothing can reach 101%")
        self.assertEqual(body["match"]["threshold"], 1.01)

    def test_analyse_without_a_connector_still_works(self):
        response = client.post("/analyse", files={"image": ("p.jpg", photo("2"), "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("match", response.json())


class SecurityTests(unittest.TestCase):
    def test_an_api_key_is_required_everywhere_when_one_is_set(self):
        with mock.patch.object(server, "API_KEY", "s3cret"):
            self.assertEqual(client.get("/references").status_code, 401)
            self.assertEqual(client.get("/references/1/thumbnail").status_code, 401)
            self.assertEqual(client.get("/references/1/image").status_code, 401)
            self.assertEqual(post_match(photo("2"), "2").status_code, 401)
            ok = {"X-API-Key": "s3cret"}
            self.assertEqual(client.get("/references", headers=ok).status_code, 200)
            self.assertEqual(client.get("/references/1/thumbnail", headers=ok).status_code, 200)
            self.assertEqual(client.get("/references", headers={"X-API-Key": "nope"}).status_code, 401)
            self.assertEqual(client.get("/health").status_code, 200, "health stays open")

    def test_only_localhost_pages_may_call_it_from_a_browser(self):
        def preflight(origin):
            return client.options(
                "/match",
                headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
            )

        self.assertEqual(
            preflight("http://localhost:63462").headers.get("access-control-allow-origin"),
            "http://localhost:63462",
        )
        self.assertNotIn("access-control-allow-origin", preflight("http://evil.example").headers)


class ThresholdSettingTests(unittest.TestCase):
    def test_default_when_not_set(self):
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("MATCH_THRESHOLD", None)
            self.assertEqual(server._threshold_from_env(), 0.70)

    def test_a_valid_value_is_used(self):
        with mock.patch.dict(os.environ, {"MATCH_THRESHOLD": "0.55"}):
            self.assertEqual(server._threshold_from_env(), 0.55)

    def test_a_bad_value_stops_the_server_with_a_clear_message(self):
        for bad in ("abc", "1.5", "-0.1", ""):
            with mock.patch.dict(os.environ, {"MATCH_THRESHOLD": bad}):
                with self.assertRaises(SystemExit) as raised:
                    server._threshold_from_env()
                self.assertIn("MATCH_THRESHOLD", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
