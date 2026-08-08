"""Hand gesture recognition from triangulated 3D landmarks.

Works on world-space points rather than image coordinates, so a gesture
reads the same whichever camera happens to see it and however the hand is
rotated. Every threshold is expressed as a fraction of the hand's own
scale (wrist to middle knuckle, ~90 mm on an adult), so it does not need
tuning per person or per camera distance.

Classification is deliberately a set of independent predicates rather than
a decision tree: "index extended" and "thumb touching ring" are separate
questions, and a frame can legitimately answer yes to several. The caller
decides what to do with an ambiguous frame.

Landmark order is MediaPipe's, see detectors.HAND_LANDMARKS.
"""

from collections import deque

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)            # cmc, mcp, ip, tip
INDEX = (5, 6, 7, 8)            # mcp, pip, dip, tip
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}

# name -> label shown in the UI, in the order the indicators appear
GESTURES = [
    ("flat_splayed", "Flat / splayed"),
    ("fist", "Fist"),
    ("point", "Pointing"),
    ("middle_finger", "Middle finger"),
    ("thumb_index", "Thumb + index"),
    ("thumb_middle", "Thumb + middle"),
    ("thumb_ring", "Thumb + ring"),
    ("thumb_pinky", "Thumb + pinky"),
    ("vulcan", "Vulcan"),
]


def _d(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def classify(points):
    """points: 21 world-space [x, y, z], any of which may be None.

    Returns {"scale_mm", "extended": {...}, "gestures": [names]} or None
    when too much of the hand is missing to judge.
    """
    if not points or len(points) < 21:
        return None
    P = [None if p is None else np.asarray(p, float) for p in points]
    need = [WRIST, INDEX[0], MIDDLE[0], RING[0], PINKY[0], THUMB[3],
            INDEX[3], MIDDLE[3], RING[3], PINKY[3]]
    if any(P[i] is None for i in need):
        return None

    # hand scale: wrist to middle knuckle. Every threshold below is a
    # fraction of this, so nothing depends on hand size or distance.
    scale = _d(P[WRIST], P[MIDDLE[0]])
    if scale < 1e-6:
        return None

    ext = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        if P[pip] is None:
            # fall back to the knuckle: a curled tip sits closer to the
            # wrist than its own knuckle, an extended one further
            ext[name] = _d(P[tip], P[WRIST]) > _d(P[mcp], P[WRIST]) * 1.35
            continue
        # straightness: an extended finger's tip is far from its knuckle
        # relative to the length of the bones getting there
        chain = _d(P[mcp], P[pip]) + _d(P[pip], P[tip])
        ext[name] = chain > 1e-6 and _d(P[mcp], P[tip]) / chain > 0.82
    # the thumb folds across the palm rather than curling, so it is judged
    # by how far its tip sits from the index knuckle
    ext["thumb"] = _d(P[THUMB[3]], P[INDEX[0]]) > 0.62 * scale

    def touch(a, b):
        return _d(P[a], P[b]) < 0.30 * scale

    def gap(a, b):
        return _d(P[a], P[b]) / scale

    fingers_out = [f for f in FINGERS if ext[f]]
    n_out = len(fingers_out)
    found = []

    # splayed: all four out AND spread apart at the tips
    spread = min(gap(INDEX[3], MIDDLE[3]), gap(MIDDLE[3], RING[3]),
                 gap(RING[3], PINKY[3])) if n_out == 4 else 0.0
    if n_out == 4 and ext["thumb"] and spread > 0.33:
        found.append("flat_splayed")
    if n_out == 0 and not ext["thumb"]:
        found.append("fist")
    if fingers_out == ["index"]:
        found.append("point")
    if fingers_out == ["middle"]:
        found.append("middle_finger")

    # thumb-to-fingertip pinches. Only the nearest one counts, so a
    # loosely closed hand does not light up four indicators at once.
    pinches = {"thumb_index": INDEX[3], "thumb_middle": MIDDLE[3],
               "thumb_ring": RING[3], "thumb_pinky": PINKY[3]}
    near = [(k, _d(P[THUMB[3]], t)) for k, t in pinches.items()
            if touch(THUMB[3], t)]
    if near:
        pinch = min(near, key=lambda kv: kv[1])[0]
        found.append(pinch)
        # a thumb-to-index pinch is not a point, whatever the finger
        # predicate says in isolation
        for clash in ({"thumb_index": "point",
                       "thumb_middle": "middle_finger"}.get(pinch),):
            if clash in found:
                found.remove(clash)

    # Vulcan: all four out, index+middle together, ring+pinky together,
    # and a clear gap in the middle. The middle gap must beat both pairs.
    if n_out == 4:
        im, rp, mr = (gap(INDEX[3], MIDDLE[3]), gap(RING[3], PINKY[3]),
                      gap(MIDDLE[3], RING[3]))
        # the ratios are what separate this from a splayed hand; the
        # absolute gap only rules out four fingers held together
        if mr > 0.38 and mr > im * 1.5 and mr > rp * 1.5:
            found.append("vulcan")

    return {"scale_mm": round(scale, 1),
            "extended": ext,
            "gestures": found}


class GestureVoter:
    """Majority vote over a short window, per hand.

    Landmark noise flips a borderline predicate frame to frame; requiring
    a gesture to hold for most of a short window trades a little latency
    for a steady indicator. Defaults to 3 of the last 4 frames.
    """

    def __init__(self, window=4, need=3):
        self.window = int(window)
        self.need = int(need)
        self._hist = {}

    def update(self, hand_id, gestures):
        h = self._hist.get(hand_id)
        if h is None:
            h = self._hist[hand_id] = deque(maxlen=self.window)
        h.append(set(gestures or []))
        counts = {}
        for frame in h:
            for g in frame:
                counts[g] = counts.get(g, 0) + 1
        return sorted(g for g, c in counts.items() if c >= self.need)

    def forget(self, keep_ids):
        for k in [k for k in self._hist if k not in keep_ids]:
            self._hist.pop(k, None)
