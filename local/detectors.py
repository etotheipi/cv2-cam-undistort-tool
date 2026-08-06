"""Pluggable object detectors for the Live Tracking tab.

A detector takes one camera frame and returns a flat list of observations
in *image* coordinates. Everything downstream — triangulation, the 3D view,
the item list — works off that one shape, so adding a detector never
touches the tracker or the UI.

    detect(frame_bgr, meta) -> [obs, ...]

    obs = {
      "kind":   detector key, e.g. "aruco" | "hands"
      "id":     stable-ish identity within this frame ("tag:5", "hand:0")
      "label":  human-readable, shown in the 3D view
      "points": [[x, y], ...]      image px, the geometry to triangulate
      "names":  ["wrist", ...]     optional, one per point
      "bbox":   [x, y, w, h]       optional, image px
      "score":  0..1               optional confidence
      "pose":   {"T_cam_obj": 4x4} optional; only detectors that recover a
                                   full single-view pose set this (ArUco
                                   does, landmark models do not)
    }

`points` is the contract that matters: same-length, same-order across
cameras for a given `id`, because multi-view fusion matches point k in one
camera to point k in another. Tags give 4 corners in a fixed winding;
hands give 21 landmarks in MediaPipe's fixed order.

Detectors declare whether they need a GPU. The registry reports what is
importable on this machine so the UI can show why something is unavailable
rather than silently omitting it.
"""

import os
import threading
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/"
                  "hand_landmarker.task")


def _ensure_model(url, filename, timeout=60):
    """Canned model, fetched once and cached next to the code. Returns the
    local path, or raises with something the UI can show."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, \
                open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)       # never leave a truncated model behind
    except Exception as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(
            f"could not download the model ({e.__class__.__name__}); "
            f"fetch {url} manually into {MODEL_DIR}") from e
    return path

# MediaPipe's hand landmark order — index k here is index k in "points".
HAND_LANDMARKS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
# bone connections, for drawing a skeleton in the 3D view
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class Detector:
    """Base class. Subclasses set the metadata and implement detect().

    detect() is called concurrently, one thread per camera. Detectors that
    keep per-camera state must therefore keep it *per camera* and guard it
    with `_cam_lock(node)` — a single detector-wide lock would serialise
    every camera and throw away the parallelism entirely.
    """

    key = "base"
    name = "Base"
    description = ""
    requires_gpu = False
    point_names = None
    edges = None
    install_hint = None

    def __init__(self):
        self._guard = threading.Lock()      # guards lazy per-camera setup
        self._cam_locks = {}
        self._cam_state = {}

    def _cam_lock(self, node):
        with self._guard:
            lk = self._cam_locks.get(node)
            if lk is None:
                lk = self._cam_locks[node] = threading.Lock()
            return lk

    def _cam_obj(self, node, factory):
        """Lazily build and cache one worker object per camera. Held
        outside the per-camera lock so a slow first build on one camera
        doesn't stall the others."""
        with self._guard:
            obj = self._cam_state.get(node)
            if obj is not None:
                return obj
        obj = factory()
        with self._guard:
            return self._cam_state.setdefault(node, obj)

    def available(self):
        """(ok, reason) — whether this detector can actually run here."""
        return True, None

    def configure(self, **kw):
        pass

    def detect(self, frame, meta):
        raise NotImplementedError

    def close(self):
        pass

    def info(self):
        ok, reason = self.available()
        return {"key": self.key, "name": self.name,
                "description": self.description,
                "requires_gpu": self.requires_gpu,
                "available": bool(ok), "reason": reason,
                "install_hint": self.install_hint,
                "point_names": self.point_names,
                "edges": self.edges}


class ArucoDetector(Detector):
    """AprilTag 36h11 tags — the same detector the pose tab uses.

    A tag is planar and its corner order is known, so one camera already
    yields a full 6-DoF pose; multi-view fusion improves it but is not
    required. Points are the 4 corners in detector order (TL, TR, BR, BL).
    """

    key = "aruco"
    name = "ArUco tags (AprilTag 36h11)"
    description = ("Fiducial markers. One camera is enough for a full pose; "
                   "extra views tighten it. Fastest option, no model to load.")
    requires_gpu = False
    point_names = ["c0", "c1", "c2", "c3"]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def __init__(self, marker_mm=40.0):
        super().__init__()
        self.marker_mm = float(marker_mm)

    def configure(self, marker_mm=None, **kw):
        if marker_mm:
            self.marker_mm = float(marker_mm)

    def _make(self):
        return cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))

    def detect(self, frame, meta):
        node = meta.get("node", 0)
        # one detector per camera: cv2's ArucoDetector is not documented as
        # safe to call concurrently, and these are cheap to hold
        det = self._cam_obj(node, self._make)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._cam_lock(node):
            corners, ids, _rej = det.detectMarkers(gray)
        if ids is None or not len(ids):
            return []
        out = []
        for c4, tid in zip(corners, ids.ravel()):
            pts = c4.reshape(4, 2)
            out.append({
                "kind": self.key,
                "id": f"tag:{int(tid)}",
                "label": f"tag {int(tid)}",
                "points": np.round(pts, 2).tolist(),
                "names": self.point_names,
                "size_mm": self.marker_mm,
            })
        return out


class HandsDetector(Detector):
    """MediaPipe Hands — 21 landmarks per hand.

    The model returns normalized image coordinates plus a *relative* depth
    that is not metric, so unlike a tag a single view cannot place a hand
    in the world. Landmark k is the same anatomical point in every camera,
    which is exactly what multi-view triangulation needs.

    Handedness ("Left"/"Right") is the only cross-camera identity signal
    available, so it becomes part of the id. Two same-handed hands in one
    frame are disambiguated by index, which is stable within a frame but
    not guaranteed across cameras — a documented limit rather than a bug.

    The pip wheel is CPU-only on Linux; requires_gpu stays False and the
    GPU columns exist for detectors added later.
    """

    key = "hands"
    name = "Hands / fingers (MediaPipe)"
    description = ("21 landmarks per hand. Needs two or more cameras with "
                   "line of sight to place a hand in 3D — a landmark model "
                   "has no metric scale from one view.")
    requires_gpu = False
    point_names = HAND_LANDMARKS
    edges = HAND_EDGES
    install_hint = ("pip install mediapipe  (pulls opencv-contrib-python, "
                    "which replaces opencv-python); the 7.8 MB model is "
                    "downloaded once on first use")

    def __init__(self, max_hands=4, min_confidence=0.5):
        super().__init__()
        self.max_hands = int(max_hands)
        self.min_confidence = float(min_confidence)
        self._ts = {}               # per-camera monotonic video timestamp

    def available(self):
        try:
            from mediapipe.tasks.python import vision  # noqa: F401
        except Exception as e:      # ImportError, or a broken native build
            return False, f"mediapipe not installed ({e.__class__.__name__})"
        path = os.path.join(MODEL_DIR, "hand_landmarker.task")
        if not os.path.exists(path):
            return True, "model will be downloaded on first use (7.8 MB)"
        return True, None

    def configure(self, max_hands=None, min_confidence=None, **kw):
        if max_hands:
            self.max_hands = int(max_hands)
        if min_confidence:
            self.min_confidence = float(min_confidence)

    def _make(self):
        """The landmarker tracks between frames, so each camera needs its
        own — sharing one would look like a violent jump cut and destroy
        tracking on every camera at once. It also means cameras can run
        concurrently: mediapipe releases the GIL during inference, so N
        landmarkers on N threads scale nearly linearly."""
        from mediapipe.tasks.python import vision, BaseOptions
        model = _ensure_model(HAND_MODEL_URL, "hand_landmarker.task")
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence))

    def detect(self, frame, meta):
        import mediapipe as mp
        node = meta.get("node", 0)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        g = self._cam_obj(node, self._make)
        with self._cam_lock(node):   # one landmarker is not re-entrant
            # VIDEO mode demands strictly increasing timestamps per stream;
            # a wall clock can repeat a millisecond, so count instead
            ts = self._ts.get(node, 0) + 33
            self._ts[node] = ts
            res = g.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.hand_landmarks:
            return []
        handed = res.handedness or []
        out = []
        for i, lm in enumerate(res.hand_landmarks):
            side, score = "hand", None
            if i < len(handed) and handed[i]:
                cl = handed[i][0]
                side = (cl.category_name or "hand").lower()
                score = round(float(cl.score), 3)
            pts = [[round(p.x * w, 2), round(p.y * h, 2)] for p in lm]
            arr = np.array(pts)
            x0, y0 = arr.min(0)
            x1, y1 = arr.max(0)
            out.append({
                "kind": self.key,
                "id": f"hand:{side}:{i}",
                "label": f"{side} hand",
                "points": pts,
                "names": HAND_LANDMARKS,
                "bbox": [round(x0, 1), round(y0, 1),
                         round(x1 - x0, 1), round(y1 - y0, 1)],
                "score": score,
            })
        return out

    def close(self):
        with self._guard:
            objs = list(self._cam_state.values())
            self._cam_state = {}
        for g in objs:
            try:
                g.close()
            except Exception:
                pass


_CLASSES = [ArucoDetector, HandsDetector]


def registry():
    """Metadata for every known detector, importable or not."""
    out = []
    for cls in _CLASSES:
        try:
            out.append(cls().info())
        except Exception as e:
            out.append({"key": cls.key, "name": cls.name,
                        "description": cls.description,
                        "requires_gpu": cls.requires_gpu,
                        "available": False,
                        "reason": f"{e.__class__.__name__}: {e}",
                        "install_hint": cls.install_hint,
                        "point_names": cls.point_names,
                        "edges": cls.edges})
    return out


def build(keys, **cfg):
    """Instantiate the requested detectors, skipping unavailable ones.
    Returns (detectors, skipped) so the caller can say what was dropped."""
    made, skipped = [], []
    by_key = {c.key: c for c in _CLASSES}
    for k in keys or []:
        cls = by_key.get(k)
        if cls is None:
            skipped.append({"key": k, "reason": "unknown detector"})
            continue
        try:
            d = cls()
            ok, reason = d.available()
            if not ok:
                skipped.append({"key": k, "reason": reason})
                continue
            d.configure(**cfg)
            made.append(d)
        except Exception as e:
            skipped.append({"key": k, "reason": f"{e.__class__.__name__}: {e}"})
    return made, skipped
