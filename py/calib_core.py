"""Camera-calibration core, running client-side inside Pyodide.

All entry points take/return JSON strings (plus raw RGBA buffers) so the
JS <-> Python bridge stays trivial. Ported from the local Flask tool's
calibration.py — same algorithms, same output schema.
"""

import json

import cv2
import numpy as np

UNIT_TO_MM = {"mm": 1.0, "m": 1000.0, "in": 25.4, "ft": 304.8}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_corners(gray, pattern):
    """Robust corner detection: SB detector first, classic as fallback."""
    flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE | getattr(cv2, "CALIB_CB_ACCURACY", 0)
    try:
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags_sb)
    except cv2.error:
        found, corners = False, None
    if found:
        return corners.astype(np.float32)
    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.astype(np.float32)


def _rgba(buf, w, h):
    """JS Uint8ClampedArray (RGBA) -> HxWx4 numpy view."""
    data = buf.to_py()
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)


def _objp(cols, rows, square):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


# ---------------------------------------------------------------------------
# Corner detection (collection + measurement snaps)
# ---------------------------------------------------------------------------

def detect_corners(buf, w, h, cols, rows):
    gray = cv2.cvtColor(_rgba(buf, w, h), cv2.COLOR_RGBA2GRAY)
    corners = _find_corners(gray, (cols, rows))
    if corners is None:
        return json.dumps(None)
    return json.dumps([[round(float(x), 3), round(float(y), 3)]
                       for x, y in corners.reshape(-1, 2)])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(corner_sets_json, w, h, cols, rows, square):
    """corner_sets_json: [[ [x,y], ... ], ...] one set per captured view."""
    sets = json.loads(corner_sets_json)
    objp = _objp(cols, rows, square)
    n_expected = cols * rows
    img_points, keep = [], []
    for i, s in enumerate(sets):
        if len(s) == n_expected:
            img_points.append(np.asarray(s, np.float32).reshape(-1, 1, 2))
            keep.append(i)
    if len(img_points) < 5:
        return json.dumps({"error":
            f"Only {len(img_points)} usable views (need >= 5)"})
    obj_points = [objp] * len(img_points)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)
    per_view, reprojected = [], []
    for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1))))
        per_view.append(round(err, 4))
        reprojected.append(np.round(proj.reshape(-1, 2), 2).tolist())
    return json.dumps({
        "rms": round(float(rms), 4),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "image_size": [w, h],
        "per_view": per_view,
        "reprojected": reprojected,
        "used_indices": keep,
    })


# ---------------------------------------------------------------------------
# Active calibration: live undistortion
# ---------------------------------------------------------------------------

_active = {"K": None, "dist": None, "size": None, "maps": None, "key": None}


def set_active_calibration(K_json, dist_json, cw, ch):
    _active["K"] = np.array(json.loads(K_json), np.float64)
    _active["dist"] = np.array(json.loads(dist_json), np.float64)
    _active["size"] = (int(cw), int(ch))
    _active["maps"] = _active["key"] = None
    return True


def clear_active_calibration():
    _active.update({"K": None, "dist": None, "size": None,
                    "maps": None, "key": None})
    return True


def _scaled_K(w, h):
    """Calibration K scaled to the current stream resolution."""
    K = _active["K"].copy()
    cw, ch = _active["size"]
    if (cw, ch) != (w, h):
        K = np.diag([w / cw, h / ch, 1.0]) @ K
    return K


def undistort_frame(buf, w, h):
    img = _rgba(buf, w, h)
    if _active["key"] != (w, h):
        K = _scaled_K(w, h)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, _active["dist"], (w, h), 0)
        m1, m2 = cv2.initUndistortRectifyMap(
            K, _active["dist"], None, newK, (w, h), cv2.CV_16SC2)
        _active["maps"], _active["key"] = (m1, m2), (w, h)
    out = cv2.remap(img, *_active["maps"], cv2.INTER_LINEAR)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Measurement on the checkerboard plane
# ---------------------------------------------------------------------------

_meas = {}


def prepare_measure(buf, w, h, cols, rows, square, already_undistorted):
    """Detect the board in a snapped frame and fit pixel->board homography.

    Raw frames with an active calibration get their points undistorted
    first, so the homography is exact; frames snapped from the undistorted
    view (or with no calibration) use points directly.
    """
    img = _rgba(buf, w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    corners = _find_corners(gray, (cols, rows))
    _meas.clear()
    if corners is None:
        return json.dumps({"found": False})
    px = corners.reshape(-1, 2)
    use_undist = _active["K"] is not None and not already_undistorted
    if use_undist:
        K = _scaled_K(w, h)
        px_fit = cv2.undistortPoints(
            px.reshape(-1, 1, 2), K, _active["dist"], P=K).reshape(-1, 2)
    else:
        K = None
        px_fit = px
    board = _objp(cols, rows, float(square))[:, :2]
    H, _ = cv2.findHomography(px_fit, board, cv2.RANSAC, 2.0)
    if H is None:
        return json.dumps({"found": False})
    _meas.update({"H": H, "use_undist": use_undist, "K": K})
    return json.dumps({"found": True,
                       "corners": np.round(px, 2).tolist()})


def measure_points(x1, y1, x2, y2):
    if "H" not in _meas:
        return json.dumps({"error": "no board prepared"})
    pts = np.array([[x1, y1], [x2, y2]], np.float32)
    if _meas["use_undist"]:
        pts = cv2.undistortPoints(
            pts.reshape(-1, 1, 2), _meas["K"], _active["dist"],
            P=_meas["K"]).reshape(-1, 2)
    b = cv2.perspectiveTransform(
        pts.reshape(-1, 1, 2).astype(np.float64), _meas["H"]).reshape(-1, 2)
    d = float(np.linalg.norm(b[0] - b[1]))
    return json.dumps({"distance": d,
                       "p1_board": [round(float(v), 3) for v in b[0]],
                       "p2_board": [round(float(v), 3) for v in b[1]]})


# ---------------------------------------------------------------------------
# Rectified (top-down) view of the checkerboard plane
# ---------------------------------------------------------------------------

_rect = {"pixels": None}


def rectify_frame(buf, w, h, cols, rows, square, already_undistorted,
                  max_out=1600):
    """Undistort (if calibrated) and warp so the checkerboard plane is
    metrically square. Returns meta JSON (or null if no board); the pixel
    buffer is fetched separately via get_rect_pixels().

    Off-plane content is distorted by design; regions near the horizon
    would map toward infinity, so the output is clamped to a few board
    sizes around the board.
    """
    img = _rgba(buf, w, h)
    use_undist = _active["K"] is not None and not already_undistorted
    if use_undist:
        K = _scaled_K(w, h)
        im = cv2.undistort(img, K, _active["dist"])
    else:
        im = img
    gray = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
    corners = _find_corners(gray, (cols, rows))
    if corners is None:
        return json.dumps(None)
    board = _objp(cols, rows, float(square))[:, :2]
    H, _ = cv2.findHomography(corners.reshape(-1, 2), board, cv2.RANSAC, 2.0)
    if H is None:
        return json.dumps(None)

    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                           np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(img_corners, H).reshape(-1, 2)
    b_min, b_max = board.min(axis=0), board.max(axis=0)
    b_size = np.maximum(b_max - b_min, 1e-6)
    lo = np.maximum(mapped.min(axis=0), b_min - 4.0 * b_size)
    hi = np.minimum(mapped.max(axis=0), b_max + 4.0 * b_size)
    lo = np.minimum(lo, b_min - 0.5 * b_size)   # always include the board
    hi = np.maximum(hi, b_max + 0.5 * b_size)
    span = hi - lo
    s = max_out / float(max(span))              # output px per board unit
    if span[0] * s * span[1] * s > 4e6:         # cap total pixels
        s *= (4e6 / (span[0] * s * span[1] * s)) ** 0.5
    out_w, out_h = int(round(span[0] * s)), int(round(span[1] * s))
    A = np.array([[s, 0, -s * lo[0]], [0, s, -s * lo[1]], [0, 0, 1]])
    out = cv2.warpPerspective(im, A @ H, (out_w, out_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(52, 58, 66, 255))
    _rect["pixels"] = out.tobytes()
    return json.dumps({"width": out_w, "height": out_h,
                       "px_per_unit": round(s, 6),
                       "undistorted": bool(use_undist)})


def get_rect_pixels():
    return _rect["pixels"]


def cv2_version():
    return cv2.__version__
