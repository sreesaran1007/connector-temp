"""
Compare the parts found in a photo with a reference connector.

A connector is read as a list of positions in order along its length; each
position holds one part, or several parts stacked on top of each other. Two
connectors are compared with a weighted edit distance over those positions and
turned into a similarity between 0 (nothing in common) and 1 (identical).

Why a score and not exact equality: the model sometimes calls one part by a
neighbouring name (for example a "female connector" instead of a "female
connector mid"), and a slightly different crop or distance can change one
position. Exact equality would call the very same connector "different". A
cut-off on the score tolerates a few such slips but still separates different
connectors. The cut-off (DEFAULT_THRESHOLD) is a starting point to tune with
real photos.

Two further allowances, because they do not change which connector it is:
  * the order of parts stacked within one position does not matter, and
  * the photo may have been taken from either end, so it is also tried reversed.
"""

from collections import Counter
from dataclasses import dataclass

# Similarity at or above this counts as "the same connector".
DEFAULT_THRESHOLD = 0.70

MAX_DIFFERENCES = 8

Position = tuple  # sorted tuple of class names, e.g. ("female connector mid", "female pin")


def positions_of(structure):
    """The positions of a structure produced by build_structure, in reading order."""
    return [tuple(sorted(c["class"] for c in p["components"])) for p in structure["sequence"]]


def _substitution_cost(a, b):
    """0 for identical positions, 1 for nothing in common, in between when the
    two stacks share some parts (a partial slip, not a total miss)."""
    if a == b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return 1.0 - sum((ca & cb).values()) / union


def _align(reference, photo):
    """Cheapest way to turn `reference` into `photo`.

    Returns (distance, operations). Inserting or deleting a position costs 1;
    replacing one costs _substitution_cost. Each operation is a tuple
    (kind, reference_index, photo_index) with 0-based indexes.
    """
    n, m = len(reference), len(photo)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = float(i)
    for j in range(1, m + 1):
        d[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1.0,
                d[i][j - 1] + 1.0,
                d[i - 1][j - 1] + _substitution_cost(reference[i - 1], photo[j - 1]),
            )

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = _substitution_cost(reference[i - 1], photo[j - 1])
            if abs(d[i][j] - (d[i - 1][j - 1] + cost)) < 1e-9:
                if cost > 0:
                    ops.append(("different", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(d[i][j] - (d[i - 1][j] + 1.0)) < 1e-9:
            ops.append(("missing", i - 1, None))
            i -= 1
        else:
            ops.append(("extra", None, j - 1))
            j -= 1
    ops.reverse()
    return d[n][m], ops


def similarity(reference, photo):
    """0..1 similarity of two position lists, in the order given."""
    longest = max(len(reference), len(photo))
    if longest == 0:
        return 1.0
    distance, _ = _align(reference, photo)
    return max(0.0, 1.0 - distance / longest)


def _label(position):
    return " + ".join(name[:1].upper() + name[1:] for name in position)


@dataclass(frozen=True)
class MatchOutcome:
    similarity: float
    threshold: float
    matches: bool
    reversed: bool  # the photo matched better when read from the other end
    differences: list  # readable sentences; empty when identical

    def as_dict(self):
        return {
            "similarity": round(self.similarity, 3),
            "threshold": self.threshold,
            "matches": self.matches,
            "reversed": self.reversed,
            "differences": self.differences,
        }


def compare(reference, photo, threshold=DEFAULT_THRESHOLD):
    """Compare a reference connector with the connector seen in a photo.

    Both are lists of positions (see positions_of). Positions in the messages are
    numbered from 1 along the reference.
    """
    if not photo:
        return MatchOutcome(0.0, threshold, False, False, ["No connector parts were found in the photo."])

    forward, ops_forward = _align(reference, photo)
    backward, ops_backward = _align(reference, list(reversed(photo)))
    use_reversed = backward < forward
    photo_used = list(reversed(photo)) if use_reversed else photo
    distance, ops = (backward, ops_backward) if use_reversed else (forward, ops_forward)
    longest = max(len(reference), len(photo))
    score = max(0.0, 1.0 - distance / longest)

    differences = []
    for kind, ref_index, photo_index in ops:
        if kind == "different":
            differences.append(
                f"Position {ref_index + 1}: expected {_label(reference[ref_index])}, "
                f"found {_label(photo_used[photo_index])}"
            )
        elif kind == "missing":
            differences.append(
                f"Position {ref_index + 1}: expected {_label(reference[ref_index])}, "
                "but nothing was found there"
            )
        else:
            differences.append(f"Extra part found: {_label(photo_used[photo_index])} (not in the reference)")
    if len(differences) > MAX_DIFFERENCES:
        hidden = len(differences) - MAX_DIFFERENCES
        differences = differences[:MAX_DIFFERENCES] + [f"...and {hidden} more"]

    return MatchOutcome(score, threshold, score >= threshold, use_reversed, differences)
