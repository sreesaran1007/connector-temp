"""Tests for matching.py (no model needed).

Run from 07-app:  .venv\\Scripts\\python -m unittest discover -s tests -t .
"""

import unittest

from matching import (
    DEFAULT_THRESHOLD,
    MAX_DIFFERENCES,
    compare,
    positions_of,
    similarity,
)

_NAMES = {
    "F": "female connector", "Fm": "female connector mid", "Fs": "female connector sm",
    "Mm": "male connector mid", "Ms": "male connector sm", "mp": "male pin",
    "fp": "female pin", "MX": "mixed", "E": "empty",
}


def P(text):
    """'mp | Fm/Fm | fp' -> list of positions (sorted tuples of class names)."""
    return [tuple(sorted(_NAMES[a] for a in position.split("/"))) for position in text.split(" | ")]


# What the model really returned for the four reference photos.
REFERENCES = {
    "1": P("F | F | Ms | Ms | Ms | F | fp"),
    "2": P("mp | Fm/Fm | Fs | Fs | Fs | Fm/F | fp"),
    "3": P("mp | Fm/F | Fm/F | Fm/F | Fs | Fs | fp"),
    "4": P("mp | Mm | Mm | Ms | Ms | Mm/F | Mm/F | fp"),
}

# What the model returned for the same connectors photographed or edited
# differently (cropped, scaled, mirrored). These are the realistic cases.
REAL_VARIANTS = {
    "2 cropped": ("2", P("mp | Fm/Fm | Fs | Fs | Ms | Fm/F | E | fp")),
    "2 mirrored": ("2", P("fp | Fm/F | Fs | Fs | Fs | Fm/Fm | mp")),
    "3 scaled": ("3", P("mp/mp | Fm/F | Fs | Fm/F | Fm/F | Fs | Fs | fp")),
    "3 mirrored": ("3", P("fp | Fs | Fs | F | Fm/F | Fs | Fm/F | mp")),
    "4 cropped": ("4", P("mp | Mm | Mm | Ms | Ms | F | Mm/F | fp")),
    "4 mirrored": ("4", P("fp | Mm/F | Mm/F | Ms | Ms | Mm | Mm | mp")),
    "1 mirrored": ("1", P("mp | F | F | Ms | Ms | Ms | F | fp")),
}


class SimilarityTests(unittest.TestCase):
    def test_identical_connectors_score_one(self):
        for ref in REFERENCES.values():
            self.assertEqual(similarity(ref, ref), 1.0)

    def test_two_empty_lists_are_identical(self):
        self.assertEqual(similarity([], []), 1.0)

    def test_nothing_in_common_scores_zero(self):
        self.assertEqual(similarity(P("mp | mp"), P("fp | fp")), 0.0)

    def test_partly_shared_stack_gets_partial_credit(self):
        # ("a","b") vs ("a","c") share 1 of 3 distinct parts: cost 1 - 1/3.
        a, b = [("a", "b")], [("a", "c")]
        self.assertAlmostEqual(similarity(a, b), 1 / 3, places=6)

    def test_is_symmetric(self):
        for x in REFERENCES.values():
            for y in REFERENCES.values():
                self.assertAlmostEqual(similarity(x, y), similarity(y, x), places=9)

    def test_a_missing_position_costs_one_over_the_longer_length(self):
        ref = P("mp | Fs | Fs | fp")
        photo = P("mp | Fs | fp")
        self.assertAlmostEqual(similarity(ref, photo), 1 - 1 / 4)


class PositionsOfTests(unittest.TestCase):
    def test_order_inside_a_stack_does_not_matter(self):
        def structure(*classes):
            return {"sequence": [{"components": [{"class": c} for c in classes]}]}

        self.assertEqual(
            positions_of(structure("female pin", "female connector mid")),
            positions_of(structure("female connector mid", "female pin")),
        )
        self.assertEqual(
            positions_of(structure("b", "a")), [("a", "b")], "sorted, as a tuple"
        )

    def test_no_sequence_gives_no_positions(self):
        self.assertEqual(positions_of({"sequence": []}), [])


class CompareTests(unittest.TestCase):
    def test_identical_is_a_match_with_nothing_different(self):
        outcome = compare(REFERENCES["2"], REFERENCES["2"])
        self.assertTrue(outcome.matches)
        self.assertEqual(outcome.similarity, 1.0)
        self.assertEqual(outcome.differences, [])
        self.assertFalse(outcome.reversed)
        self.assertEqual(outcome.threshold, DEFAULT_THRESHOLD)

    def test_each_reference_matches_only_itself(self):
        for a, ref_a in REFERENCES.items():
            for b, ref_b in REFERENCES.items():
                self.assertEqual(compare(ref_a, ref_b).matches, a == b, f"{a} vs {b}")

    def test_the_most_alike_pair_is_still_clearly_below_the_cut_off(self):
        scores = [
            compare(REFERENCES[a], REFERENCES[b]).similarity
            for a in REFERENCES for b in REFERENCES if a != b
        ]
        self.assertLess(max(scores), DEFAULT_THRESHOLD - 0.15)

    def test_realistic_variants_match_their_own_connector_and_no_other(self):
        for name, (own, signature) in REAL_VARIANTS.items():
            for ref_id, ref in REFERENCES.items():
                self.assertEqual(
                    compare(ref, signature).matches, ref_id == own, f"{name} vs {ref_id}"
                )

    def test_a_photo_taken_from_the_other_end_still_matches(self):
        outcome = compare(REFERENCES["3"], list(reversed(REFERENCES["3"])))
        self.assertTrue(outcome.matches)
        self.assertTrue(outcome.reversed)
        self.assertEqual(outcome.similarity, 1.0)

    def test_a_connector_read_forwards_is_not_flagged_as_reversed(self):
        self.assertFalse(compare(REFERENCES["4"], REFERENCES["4"]).reversed)

    def test_a_wrong_part_is_named_in_plain_words(self):
        photo = P("mp | Fm/Fm | Fs | Fs | Ms | Fm/F | fp")  # 5th: Ms instead of Fs
        outcome = compare(REFERENCES["2"], photo)
        self.assertEqual(
            outcome.differences,
            ["Position 5: expected Female connector sm, found Male connector sm"],
        )

    def test_a_missing_part_is_reported(self):
        photo = P("mp | Fm/Fm | Fs | Fs | Fm/F | fp")  # one Fs fewer
        outcome = compare(REFERENCES["2"], photo)
        self.assertEqual(len(outcome.differences), 1)
        self.assertIn("nothing was found there", outcome.differences[0])
        self.assertIn("Female connector sm", outcome.differences[0])

    def test_an_extra_part_is_reported(self):
        photo = P("mp | Fm/Fm | Fs | Fs | Fs | Fm/F | E | fp")
        outcome = compare(REFERENCES["2"], photo)
        self.assertEqual(outcome.differences, ["Extra part found: Empty (not in the reference)"])

    def test_stacked_parts_are_joined_with_a_plus(self):
        photo = P("mp | Fm/Mm | Fs | Fs | Fs | Fm/F | fp")
        outcome = compare(REFERENCES["2"], photo)
        self.assertIn("Female connector mid + Male connector mid", outcome.differences[0])

    def test_no_parts_in_the_photo_is_never_a_match(self):
        outcome = compare(REFERENCES["1"], [])
        self.assertFalse(outcome.matches)
        self.assertEqual(outcome.similarity, 0.0)
        self.assertEqual(outcome.differences, ["No connector parts were found in the photo."])

    def test_reaching_the_threshold_exactly_is_a_match(self):
        photo = REAL_VARIANTS["2 cropped"][1]
        score = compare(REFERENCES["2"], photo).similarity
        self.assertTrue(compare(REFERENCES["2"], photo, threshold=score).matches)
        self.assertFalse(compare(REFERENCES["2"], photo, threshold=score + 0.001).matches)

    def test_the_threshold_decides_the_verdict(self):
        photo = REAL_VARIANTS["2 cropped"][1]
        self.assertTrue(compare(REFERENCES["2"], photo, threshold=0.5).matches)
        self.assertFalse(compare(REFERENCES["2"], photo, threshold=0.95).matches)
        self.assertEqual(compare(REFERENCES["2"], photo, threshold=0.95).threshold, 0.95)

    def test_a_long_list_of_differences_is_capped(self):
        ref = [("female connector",)] * 20
        photo = [("male connector sm",)] * 20
        differences = compare(ref, photo, threshold=0.0).differences
        self.assertEqual(len(differences), MAX_DIFFERENCES + 1)
        self.assertEqual(differences[-1], f"...and {20 - MAX_DIFFERENCES} more")

    def test_as_dict_is_ready_for_json(self):
        data = compare(REFERENCES["2"], REAL_VARIANTS["2 cropped"][1]).as_dict()
        self.assertEqual(
            set(data), {"similarity", "threshold", "matches", "reversed", "differences"}
        )
        self.assertEqual(data["similarity"], round(data["similarity"], 3))
        self.assertIsInstance(data["matches"], bool)


if __name__ == "__main__":
    unittest.main()
