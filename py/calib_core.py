"""Camera-calibration core, running client-side inside Pyodide.

All entry points take/return JSON strings (plus raw RGBA buffers) so the
JS <-> Python bridge stays trivial. Ported from the local Flask tool's
calibration.py — same algorithms, same output schema.
"""

import json
import math

import cv2
import numpy as np

UNIT_TO_MM = {"mm": 1.0, "m": 1000.0, "in": 25.4, "ft": 304.8}


def angular_block(fx, fy, w, h):
    """Angular metrics for the pinhole model (valid on images undistorted
    with this camera_matrix). deg/px is only constant near the center: a
    pinhole projects x = f*tan(theta), so per-pixel angle falls off as
    cos^2(theta) off-axis."""
    fov_h = 2 * math.degrees(math.atan(w / (2 * fx)))
    fov_v = 2 * math.degrees(math.atan(h / (2 * fy)))
    fov_d = 2 * math.degrees(math.atan(math.hypot(w / (2 * fx), h / (2 * fy))))
    return {
        "applies_to": ("images undistorted with this camera_matrix "
                       "(plain cv2.undistort); if undistorting with a "
                       "different new_camera_matrix, use its fx/fy instead"),
        "radians_per_pixel_at_center": {"x": 1.0 / fx, "y": 1.0 / fy},
        "degrees_per_pixel_at_center": {"x": math.degrees(1.0 / fx),
                                        "y": math.degrees(1.0 / fy)},
        "fov_degrees": {"horizontal": round(fov_h, 4),
                        "vertical": round(fov_v, 4),
                        "diagonal": round(fov_d, 4)},
        "exact_formula": ("theta_x = atan((u - cx)/fx); per-pixel angle = "
                          "cos^2(theta)/f; ray direction = inv(K) @ [u, v, 1]"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgba(buf, w, h):
    """JS Uint8ClampedArray (RGBA) -> HxWx4 numpy view."""
    data = buf.to_py()
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)


_det_cache = {}


def _get_charuco(sx, sy, square_mm, marker_mm):
    """(board, CharucoDetector) for these params, cached."""
    key = (int(sx), int(sy), float(square_mm), float(marker_mm))
    if key not in _det_cache:
        board = _charuco_board(*key)
        _det_cache[key] = (board, cv2.aruco.CharucoDetector(board))
    return _det_cache[key]


MIN_CHARUCO_CORNERS = 8      # skip views with fewer interpolated corners


def _detect(gray, sx, sy, square_mm, marker_mm):
    """-> (corners Nx2 float32, ids N int32) or (None, None)."""
    board, det = _get_charuco(sx, sy, square_mm, marker_mm)
    corners, ids, _mk_c, _mk_ids = det.detectBoard(gray)
    if ids is None or len(ids) < MIN_CHARUCO_CORNERS:
        return None, None
    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1)


# ---------------------------------------------------------------------------
# Detection (collection + measurement snaps)
# ---------------------------------------------------------------------------

def detect_charuco(buf, w, h, sx, sy, square_mm, marker_mm):
    """ChArUco corners in one frame. Partial board views are fine — each
    corner carries its id, so any >= MIN_CHARUCO_CORNERS subset is usable."""
    gray = cv2.cvtColor(_rgba(buf, w, h), cv2.COLOR_RGBA2GRAY)
    corners, ids = _detect(gray, sx, sy, square_mm, marker_mm)
    if corners is None:
        return json.dumps(None)
    return json.dumps({
        "corners": [[round(float(x), 3), round(float(y), 3)]
                    for x, y in corners],
        "ids": [int(i) for i in ids],
        "n": int(len(ids)),
    })


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_charuco(views_json, w, h, sx, sy, square_mm, marker_mm):
    """views_json: [{"corners": [[x,y],...], "ids": [...]}, ...]."""
    views = json.loads(views_json)
    board, _ = _get_charuco(sx, sy, square_mm, marker_mm)
    obj_points, img_points, keep = [], [], []
    seen, duplicates = set(), 0
    for i, v in enumerate(views):
        c = np.asarray(v.get("corners") or [], np.float32).reshape(-1, 1, 2)
        ids = np.asarray(v.get("ids") or [], np.int32).reshape(-1, 1)
        if len(ids) < MIN_CHARUCO_CORNERS or len(ids) != len(c):
            continue
        # identical corner sets = duplicate frames (frozen stream); a
        # degenerate solve on N copies of one view must never happen
        key = (c.tobytes(), ids.tobytes())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        op, ip = board.matchImagePoints(c, ids)
        if op is None or len(op) < MIN_CHARUCO_CORNERS:
            continue
        obj_points.append(op.astype(np.float32))
        img_points.append(ip.astype(np.float32))
        keep.append(i)
    if len(img_points) < 5:
        msg = f"Only {len(img_points)} usable views (need >= 5)"
        if duplicates:
            msg += (f" — {duplicates} duplicate frames discarded; "
                    "the camera stream may have been frozen")
        return json.dumps({"error": msg})
    (rms, K, dist, rvecs, tvecs, std_int, _std_ext,
     per_view_err) = cv2.calibrateCameraExtended(
        obj_points, img_points, (w, h), None, None)
    per_view, reprojected = [], []
    for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1))))
        per_view.append(round(err, 4))
        # matchImagePoints preserves detection order, so these align with
        # each view's stored corners for the overlay artifact
        reprojected.append(np.round(proj.reshape(-1, 2), 2).tolist())
    return json.dumps({
        "rms": round(float(rms), 4),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "image_size": [w, h],
        "duplicate_views_removed": duplicates,
        "per_view": per_view,
        "per_view_corners": [int(len(ip)) for ip in img_points],
        "reprojected": reprojected,
        "used_indices": keep,
        "std_intrinsics": [round(float(v), 6)
                           for v in np.ravel(std_int)[:9]],
        "angular": angular_block(float(K[0, 0]), float(K[1, 1]), w, h),
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
# Measurement on the board plane (+ camera<->board pose)
# ---------------------------------------------------------------------------

_meas = {}


def _euler_zyx_deg(R):
    """R (board->camera) as yaw/pitch/roll, ZYX convention, degrees."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                       # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return [round(math.degrees(a), 2) for a in (yaw, pitch, roll)]


def _solve_pose(board, corners, ids, K, dist, sx, sy, square_mm):
    """solvePnP of the detected corners -> board pose in the camera frame."""
    op, ip = board.matchImagePoints(
        corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
    if op is None or len(op) < MIN_CHARUCO_CORNERS:
        return None
    ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    center = np.array([sx * square_mm / 2.0, sy * square_mm / 2.0, 0.0])
    c_cam = (R @ center + tvec.ravel())          # board center, camera frame
    normal = R @ np.array([0.0, 0.0, 1.0])       # board plane normal
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    off_axis = math.degrees(math.atan2(
        math.hypot(float(c_cam[0]), float(c_cam[1])), float(c_cam[2])))
    proj, _ = cv2.projectPoints(op, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(
        np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1))))
    yaw, pitch, roll = _euler_zyx_deg(R)
    return {
        "distance_mm": round(float(np.linalg.norm(c_cam)), 1),
        "position_mm": [round(float(v), 1) for v in c_cam],
        "tilt_deg": round(tilt, 2),          # 0 = board faces the camera
        "off_axis_deg": round(off_axis, 2),  # board center off optical axis
        "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
        "n_corners": int(len(op)),
        "reproj_rms_px": round(rms, 3),
        "convention": ("camera frame: +x right, +y down, +z forward; "
                       "T_camera_board via solvePnP"),
    }


def prepare_measure(buf, w, h, sx, sy, square_mm, marker_mm,
                    already_undistorted):
    """Detect the ChArUco board in a snapped frame, fit the pixel->board
    homography for plane measurement, and (with an active calibration)
    solve the full camera<->board pose.

    Raw frames with an active calibration get their points undistorted
    first, so the homography is exact; frames snapped from the undistorted
    view (or with no calibration) use points directly.
    """
    img = _rgba(buf, w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    board, _det_obj = _get_charuco(sx, sy, square_mm, marker_mm)
    corners, ids = _detect(gray, sx, sy, square_mm, marker_mm)
    _meas.clear()
    if corners is None:
        return json.dumps({"found": False})
    use_undist = _active["K"] is not None and not already_undistorted
    if use_undist:
        K = _scaled_K(w, h)
        px_fit = cv2.undistortPoints(
            corners.reshape(-1, 1, 2), K, _active["dist"],
            P=K).reshape(-1, 2)
    else:
        K = None
        px_fit = corners
    all_bpts = board.getChessboardCorners()          # (N,3) in mm
    board_xy = all_bpts[ids][:, :2].astype(np.float64)
    H, _ = cv2.findHomography(px_fit, board_xy, cv2.RANSAC, 2.0)
    if H is None:
        return json.dumps({"found": False})
    _meas.update({"H": H, "use_undist": use_undist, "K": K})
    pose = None
    if _active["K"] is not None:
        if already_undistorted:
            # undistort_frame remaps into the alpha=0 optimal matrix, so
            # that is the effective K of an undistorted snap
            K0 = _scaled_K(w, h)
            pK, _ = cv2.getOptimalNewCameraMatrix(
                K0, _active["dist"], (w, h), 0)
            pdist = np.zeros(5)
        else:
            pK, pdist = _scaled_K(w, h), _active["dist"]
        pose = _solve_pose(board, corners, ids, pK, pdist,
                           sx, sy, float(square_mm))
    return json.dumps({"found": True,
                       "n": int(len(ids)),
                       "corners": np.round(corners, 2).tolist(),
                       "pose": pose})


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
# Rectified (top-down) view of the board plane
# ---------------------------------------------------------------------------

_rect = {"views": []}


def rectify_views(buf, w, h, sx, sy, square_mm, marker_mm,
                  already_undistorted, max_side=1600):
    """Undistort (if calibrated) and warp so the ChArUco board plane is
    metrically square, at up to 4 zoom levels.

    Zoom levels are defined by N = (view diagonal) / (board diagonal):
    1/6, then 1/18 and 1/50 only while the previous level still crops the
    original image, and finally a best-effort everything-included view.
    If the plane's horizon crosses the frame, full coverage is impossible
    (pixels map toward infinity); that last view is clamped and flagged.

    Returns meta JSON {undistorted, views: [...]} or null if no board.
    Pixel buffers are fetched per-view via get_rect_pixels(i).
    """
    img = _rgba(buf, w, h)
    use_undist = _active["K"] is not None and not already_undistorted
    if use_undist:
        im = cv2.undistort(img, _scaled_K(w, h), _active["dist"])
    else:
        im = img
    gray = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
    charuco, _det_obj = _get_charuco(sx, sy, square_mm, marker_mm)
    corners, ids = _detect(gray, sx, sy, square_mm, marker_mm)
    _rect["views"] = []
    if corners is None:
        return json.dumps(None)
    all_bpts = charuco.getChessboardCorners()
    board = all_bpts[:, :2].astype(np.float64)       # full extent for framing
    board_xy = all_bpts[ids][:, :2].astype(np.float64)
    H, _ = cv2.findHomography(corners, board_xy, cv2.RANSAC, 2.0)
    if H is None:
        return json.dumps(None)

    b_min, b_max = board.min(0), board.max(0)
    center = (b_min + b_max) / 2
    b_diag = float(np.hypot(*(b_max - b_min)))

    # Keep the original image orientation: corner ordering has a 180-degree
    # ambiguity, so the board axes may point against the image axes. Probe
    # which way each board axis runs in image space and flip to match.
    Hinv = np.linalg.inv(H)
    def _img_of(b):
        p = Hinv @ np.array([b[0], b[1], 1.0])
        return p[:2] / p[2]
    eps = max(b_diag, 1.0) * 0.05
    origin = _img_of(center)
    ix = _img_of(center + np.array([eps, 0])) - origin
    iy = _img_of(center + np.array([0, eps])) - origin
    flip = np.eye(3)
    if ix[0] < 0:
        flip[0, 0], flip[0, 2] = -1.0, 2 * center[0]
    if iy[1] < 0:
        flip[1, 1], flip[1, 2] = -1.0, 2 * center[1]
    H = flip @ H          # board bbox is symmetric about center: unchanged

    # image corners on the board plane; homogeneous w<=0 => at/behind horizon
    pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    ph = (H @ np.vstack([pts.T, np.ones(4)])).T
    valid = ph[:, 2] > 1e-9
    unbounded = not bool(np.all(valid))
    mapped = ph[valid, :2] / ph[valid, 2:3]
    img_lo = np.minimum(mapped.min(0), b_min)
    img_hi = np.maximum(mapped.max(0), b_max)

    def render(lo, hi):
        span = np.maximum(hi - lo, 1e-6)
        s = max_side / float(max(span))
        if span[0] * span[1] * s * s > 4e6:
            s *= (4e6 / (span[0] * span[1] * s * s)) ** 0.5
        ow = max(int(round(span[0] * s)), 16)
        oh = max(int(round(span[1] * s)), 16)
        A = np.array([[s, 0, -s * lo[0]], [0, s, -s * lo[1]], [0, 0, 1]])
        out = cv2.warpPerspective(im, A @ H, (ow, oh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(52, 58, 66, 255))
        return out, s, ow, oh

    def add_view(lo, hi, name, warning, covered, clipped):
        out, s, ow, oh = render(lo, hi)
        _rect["views"].append(out.tobytes())
        return {"width": ow, "height": oh, "px_per_unit": round(s, 6),
                "lo": [float(lo[0]), float(lo[1])], "name": name,
                "warning": warning, "covers_full_image": covered,
                "clipped": clipped}

    img_diag = float(np.hypot(w, h))
    WARN_N = 20     # beyond this zoom-out the board gets small: warn

    def board_centered(n):
        half = np.array([n * b_diag * w / img_diag,
                         n * b_diag * h / img_diag]) / 2
        return center - half, center + half

    metas = []
    presets = [(6, "close-up view"), (18, "wide view"), (50, "very wide view")]
    if unbounded:
        # full coverage impossible: fixed ladder + a clamped widest view
        for n, name in presets:
            lo, hi = board_centered(n)
            metas.append(add_view(lo, hi, name, n > WARN_N, False, False))
        cap_lo = np.maximum(img_lo, center - 40 * b_diag)
        cap_hi = np.minimum(img_hi, center + 40 * b_diag)
        pad = 0.02 * (cap_hi - cap_lo)
        metas.append(add_view(cap_lo - pad, cap_hi + pad,
                              "widest (clipped at horizon)", True, False, True))
    else:
        # minimal aspect-preserving zoom that fits every original pixel,
        # centered on the mapped-image bounding box
        span = img_hi - img_lo
        n_fit = float(1.02 * max(span[0] * img_diag / (b_diag * w),
                                 span[1] * img_diag / (b_diag * h)))
        # keep only presets meaningfully tighter than the fit view
        levels = [(n, name) for n, name in presets if n <= n_fit / 1.2]
        if not levels:
            # everything already fits at (or tighter than) the base zoom
            n = max(presets[0][0], n_fit)
            lo, hi = board_centered(n)
            metas.append(add_view(lo, hi, presets[0][1], False, True, False))
        else:
            for n, name in levels:
                lo, hi = board_centered(n)
                metas.append(add_view(lo, hi, name, n > WARN_N, False, False))
            fit_c = (img_lo + img_hi) / 2
            half = np.array([n_fit * b_diag * w / img_diag,
                             n_fit * b_diag * h / img_diag]) / 2
            metas.append(add_view(fit_c - half, fit_c + half, "full scene",
                                  n_fit > WARN_N, True, False))
    return json.dumps({"undistorted": bool(use_undist), "views": metas})


def get_rect_pixels(i):
    return _rect["views"][int(i)]


def undistort_once(buf, w, h, K_json, dist_json, cal_w, cal_h):
    """Stateless undistortion for one frame (does not touch the active
    calibration). Scales the camera matrix if the frame size differs from
    the calibration size. Returns RGBA bytes."""
    img = _rgba(buf, w, h)
    K = np.array(json.loads(K_json), np.float64)
    dist = np.array(json.loads(dist_json), np.float64)
    if (int(cal_w), int(cal_h)) != (w, h):
        K = np.diag([w / cal_w, h / cal_h, 1.0]) @ K
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    out = cv2.undistort(img, K, dist, None, newK)
    return out.tobytes()


def cv2_version():
    return cv2.__version__


# --------------------------------------------------------------- ChArUco
# One dictionary for everything (calibration boards AND object tags):
# AprilTag 36h11 — 587 ids, Hamming distance 11, built into OpenCV.
CHARUCO_DICT_NAME = "DICT_APRILTAG_36h11"


def _aruco_dict():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _charuco_board(sx, sy, square_mm, marker_mm):
    return cv2.aruco.CharucoBoard(
        (int(sx), int(sy)), float(square_mm), float(marker_mm), _aruco_dict())


def charuco_board_png(sx, sy, square_mm, marker_mm, dpi, margin_mm):
    """PNG of a ChArUco board at exact physical scale: printed at 100% with
    width (sx*square_mm + 2*margin_mm), every square measures square_mm."""
    board = _charuco_board(sx, sy, square_mm, marker_mm)
    px_per_mm = float(dpi) / 25.4
    margin_px = int(round(margin_mm * px_per_mm))
    w = int(round(sx * square_mm * px_per_mm)) + 2 * margin_px
    h = int(round(sy * square_mm * px_per_mm)) + 2 * margin_px
    img = board.generateImage((w, h), marginSize=margin_px, borderBits=1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return bytes(buf.tobytes())


def charuco_board_manifest(sx, sy, square_mm, marker_mm):
    """Machine-readable board description (keep beside the printed board)."""
    board = _charuco_board(sx, sy, square_mm, marker_mm)
    return json.dumps({
        "type": "charuco",
        "dictionary": CHARUCO_DICT_NAME,
        "squares_x": int(sx),
        "squares_y": int(sy),
        "square_length_m": round(float(square_mm) / 1000.0, 6),
        "marker_length_m": round(float(marker_mm) / 1000.0, 6),
        "border_bits": 1,
        "marker_ids": [int(i) for i in board.getIds().ravel()],
        # OpenCV changed the charuco pattern convention after 4.6 for
        # even-row boards; record which one this print uses
        "legacy_pattern": False,
        "generator": "opencv " + cv2.__version__,
    })


def aruco_tag_png(tag_id, size_mm, dpi):
    """PNG of one standalone 36h11 tag. size_mm covers the black border
    (borderBits=1 quiet module included); leave white space when mounting."""
    px = int(round(float(size_mm) / 25.4 * float(dpi)))
    img = cv2.aruco.generateImageMarker(_aruco_dict(), int(tag_id), px,
                                        borderBits=1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return bytes(buf.tobytes())


def charuco_selftest(sx, sy, square_mm, marker_mm, dpi):
    """Round-trip: generate the board, then detect it — proves the printed
    pattern is decodable by the same OpenCV build and returns what it saw."""
    png = charuco_board_png(sx, sy, square_mm, marker_mm, dpi, 10)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    board = _charuco_board(sx, sy, square_mm, marker_mm)
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, mk_corners, mk_ids = det.detectBoard(img)
    return json.dumps({
        "charuco_corners": 0 if ids is None else int(len(ids)),
        "expected_corners": (int(sx) - 1) * (int(sy) - 1),
        "markers": 0 if mk_ids is None else int(len(mk_ids)),
        "expected_markers": len(board.getIds().ravel()),
    })
