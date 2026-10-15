"""Local-mode bridge server: serves the same static app as GitHub Pages,
plus /api/host/* endpoints for host-side camera access.

The browser stays the brain (all CV runs in Pyodide, identical to Pages
mode); this server only does what a page can't: enumerate USB devices with
real serials, own the cameras, and stream frames.

Run:  python -m local.server [--port 8123] [--bind 127.0.0.1]
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

try:
    from . import (cameras, detectors as detectors_mod, handcal, livetrack,
                   storage, stylus as stylus_mod, surfmap, tagsizes,
                   tracker as tracker_mod, uvcmod)
except ImportError:          # running as a plain script
    import cameras
    import detectors as detectors_mod
    import handcal
    import livetrack
    import storage
    import stylus as stylus_mod
    import surfmap
    import tagsizes
    import tracker as tracker_mod
    import uvcmod

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

app = Flask(__name__)
streams = {}                 # node -> CameraStream (several cameras at once)
# node -> (K, dist, cal_size) of the calibration the pose / live tabs were
# started with; their views (multistream ?undistort=1) are shown corrected
view_cals = {}
stream_lock = threading.Lock()
tracker = tracker_mod.Tracker(streams)
# hand-based world calibration: same streams and intrinsics as the tracker
hcal = handcal.HandCalSession(
    streams, lambda n, w, h: tracker._K_for(n, w, h),
    lambda: detectors_mod.HandsDetector(max_hands=2))
live = livetrack.LiveTracker(streams)
smap = surfmap.SurfaceMap()
live.surfmap = smap               # every stylus fix is offered to the surface map


def _save_tag_sizes(d):
    cfg = load_config()
    cfg["tag_sizes"] = d
    save_config(cfg)


# tag sizes learned from multi-camera sightings, shared by both trackers
tag_sizes = tagsizes.TagSizes(lambda: load_config().get("tag_sizes"), _save_tag_sizes)
live.tag_sizes = tag_sizes
tracker.tag_sizes = tag_sizes

_store = None
_store_cfg = None


def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"storage": {"type": "dir",
                            "dir_path": str(ROOT / "camera_cal")}}


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_store(rebuild=False):
    global _store, _store_cfg
    cfg = load_config().get("storage", {})
    if _store is None or rebuild or cfg != _store_cfg:
        _store = storage.make_storage(cfg)
        _store_cfg = cfg
    return _store


_alt_store = None
_alt_cfg = None


def alt_storage_cfg():
    """The OTHER backend: dir when the active store is S3, and vice versa."""
    cfg = load_config().get("storage", {})
    if cfg.get("type") == "s3":
        return {"type": "dir",
                "dir_path": cfg.get("dir_path") or str(ROOT / "camera_cal")}
    env = cfg.get("env_file")
    if not env and (ROOT / ".env").is_file():
        env = str(ROOT / ".env")
    return {"type": "s3", "env_file": env}


def get_alt_store(rebuild=False):
    global _alt_store, _alt_cfg
    cfg = alt_storage_cfg()
    if _alt_store is None or rebuild or cfg != _alt_cfg:
        _alt_store = storage.make_storage(cfg)
        _alt_cfg = cfg
    return _alt_store


# ---------------------------------------------------------------- static app

@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:name>")
def static_files(name):
    return send_from_directory(ROOT, name)


# ------------------------------------------------------------- host camera API

@app.get("/api/host/ping")
def ping():
    return jsonify({"mode": "host", "server": "cv2-cam-undistort-tool local bridge"})


@app.get("/api/host/cameras")
def api_cameras():
    cams = cameras.list_cameras()
    return jsonify(cams)


@app.get("/api/host/usb")
def api_usb():
    return jsonify(cameras.list_usb_tree())


@app.get("/api/host/cameras/<int:node>/details")
def api_camera_details(node):
    for cam in cameras.list_cameras():
        if cam["node"] == node:
            cam.update(cameras.camera_modes(cam["path"]))
            return jsonify(cam)
    return jsonify({"error": "camera not found"}), 404


def _cam_path(node):
    cam = next((c for c in cameras.list_cameras() if c["node"] == node), None)
    return cam["path"] if cam else None


@app.get("/api/host/cameras/<int:node>/focus")
def api_focus_get(node):
    path = _cam_path(node)
    if path is None:
        return jsonify({"error": "camera not found"}), 404
    return jsonify(cameras.focus_info(path))


@app.post("/api/host/cameras/<int:node>/focus")
def api_focus_set(node):
    path = _cam_path(node)
    if path is None:
        return jsonify({"error": "camera not found"}), 404
    body = request.get_json(force=True)
    return jsonify(cameras.focus_set(path, value=body.get("value"),
                                     auto=body.get("auto")))


@app.post("/api/host/stream/start")
def api_stream_start():
    body = request.get_json(force=True)
    node = int(body["node"])
    cam = next((c for c in cameras.list_cameras() if c["node"] == node), None)
    if cam is None:
        return jsonify({"error": "camera not found"}), 404
    try:
        with stream_lock:
            st = streams.get(node)
            if st is not None:
                st.stop()
            else:
                st = streams[node] = cameras.CameraStream()
            info = st.start(cam, int(body.get("width", 1280)),
                            int(body.get("height", 720)),
                            float(body.get("fps") or 0))
    except RuntimeError as e:
        streams.pop(node, None)
        return jsonify({"error": str(e)}), 409
    return jsonify(info)


@app.post("/api/host/stream/stop")
def api_stream_stop():
    body = request.get_json(force=True) if request.data else {}
    with stream_lock:
        if "node" in body:
            st = streams.pop(int(body["node"]), None)
            if st is not None:
                st.stop()
        else:
            for st in streams.values():
                st.stop()
            streams.clear()
    return jsonify({"ok": True})


@app.post("/api/host/reset")
def api_reset():
    """Panic button: stop tracking and every stream (releasing all USB
    bandwidth), port-reset each camera, then wait for them to re-enumerate.
    The page reloads itself afterwards, so all client state starts over."""
    live.stop()
    tracker.stop()
    with stream_lock:
        for st in streams.values():
            st.stop()
        streams.clear()
    before = len(cameras.list_cameras())
    time.sleep(0.5)                       # let released devices settle
    results = cameras.reset_usb_cameras()
    after = []
    if any(r["ok"] for r in results):
        time.sleep(2.0)                   # unbind happens async after the ioctl
        deadline = time.time() + 10
        while time.time() < deadline:
            after = cameras.list_cameras()
            if len(after) >= before:
                break
            time.sleep(0.5)
    else:
        after = cameras.list_cameras()
    body = {"cameras_before": before, "cameras_after": len(after),
            "reset": results}
    if any(r.get("permission") for r in results):
        body["fix"] = (f"echo '{cameras.UDEV_RULE}' | sudo tee "
                       f"{cameras.UDEV_RULE_PATH} && sudo udevadm control "
                       f"--reload && sudo udevadm trigger -s usb")
    return jsonify(body)


# ---------------------------------------------- uvcvideo bandwidth patch
uvc = uvcmod.UvcModule()


@app.get("/api/host/usb/reservations")
def api_usb_reservations():
    """Bytes per 125 us each USB bus has reserved for streaming cameras."""
    return jsonify({str(b): v for b, v in uvcmod.reservations().items()})


@app.get("/api/host/uvc/status")
def api_uvc_status():
    return jsonify(uvc.status())


@app.post("/api/host/uvc/build")
def api_uvc_build():
    return jsonify(uvc.build())


def _release_cameras():
    """Stop everything holding a camera: a loaded driver can't be swapped."""
    hcal.stop()
    live.stop()
    tracker.stop()
    with stream_lock:
        for st in streams.values():
            st.stop()
        streams.clear()
    time.sleep(0.5)


@app.post("/api/host/uvc/load")
def api_uvc_load():
    body = request.get_json(force=True) if request.data else {}
    before = len(cameras.list_cameras())
    _release_cameras()
    res = uvc.load(body.get("cap") or uvcmod.DEFAULT_CAP)
    res["cameras_before"] = before
    res["cameras_after"] = uvcmod.wait_for_cameras(cameras.list_cameras, before)
    return jsonify(res)


@app.post("/api/host/uvc/restore")
def api_uvc_restore():
    before = len(cameras.list_cameras())
    _release_cameras()
    res = uvc.restore()
    res["cameras_before"] = before
    res["cameras_after"] = uvcmod.wait_for_cameras(cameras.list_cameras, before)
    return jsonify(res)


@app.get("/api/host/streams")
def api_streams():
    """Active streams by node — lets the client reuse rather than restart."""
    return jsonify({str(n): dict(st.info)
                    for n, st in streams.items() if st.started})


_snapshot_lock = threading.Lock()


@app.get("/api/host/cameras/<int:node>/snapshot.jpg")
def api_camera_snapshot(node):
    """One frame from any camera, without disturbing active streams:
    reuses this camera's live stream if it has one, otherwise opens the
    device briefly."""
    width = int(request.args.get("width", 1280))
    height = int(request.args.get("height", 720))
    st = streams.get(node)
    if st is not None and st.started:
        frame, _seq, _ts = st.get_frame(0, timeout=3.0)
        if frame is None:
            return "no frame", 503
    else:
        cam = next((c for c in cameras.list_cameras() if c["node"] == node),
                   None)
        if cam is None:
            return "camera not found", 404
        with _snapshot_lock:
            cap = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
            if not cap.isOpened():
                return "camera busy or unavailable", 409
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(3):          # let auto-exposure settle
                cap.read()
            ok, frame = cap.read()
            cap.release()
        if not ok:
            return "no frame", 503
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(jpg.tobytes(), mimetype="image/jpeg")


def _mjpeg(st):
    seq = 0
    while True:
        frame, seq, _ts = st.get_frame(seq, timeout=2.0)
        if frame is None:
            if not st.started:
                break
            continue          # stream starting up or stalled; keep waiting
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpg.tobytes() + b"\r\n")


@app.get("/api/host/cameras/<int:node>/stream.mjpg")
def api_stream(node):
    st = streams.get(node)
    if st is None or not st.started:
        return "no active stream", 503
    return Response(_mjpeg(st),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def _set_view_cal(node, c):
    if c.get("K"):
        view_cals[node] = (c["K"], c.get("dist"), c.get("cal_size"))
    else:
        view_cals.pop(node, None)


def _view_maps(node, w, h, ow, oh, cache):
    """Remap tables for a camera's undistorted view, or None when it has no
    calibration (an uncalibrated camera can only be shown raw)."""
    cal = view_cals.get(node)
    if cal is None:
        return None
    key = (json.dumps(cal), w, h, ow, oh)
    hit = cache.get(node)
    if hit is None or hit[0] != key:
        hit = cache[node] = (key, tracker_mod.view_maps(*cal, w, h, ow, oh))
    return hit[1]


@app.get("/api/host/multistream")
def api_multistream():
    """All requested cameras over ONE connection (browsers cap ~6 parallel
    connections per host, so per-camera MJPEG <img> tags can't scale to 8
    cameras). Wire format per frame: "<node>,<jpeg_len>\\n" + jpeg bytes."""
    nodes = [int(x) for x in request.args.get("nodes", "").split(",")
             if x.strip().isdigit()]
    maxw = int(request.args.get("width", 480))
    quality = min(95, max(40, int(request.args.get("quality", 78))))
    fps = float(request.args.get("fps") or 0)
    min_dt = 1.0 / fps if fps > 0 else 0.0
    undistort = request.args.get("undistort") == "1"

    def gen():
        maps = {}                     # node -> (key, remap tables)
        seqs = {n: 0 for n in nodes}
        last_sent = {n: 0.0 for n in nodes}
        # A generator that never yields never sees the client go away: the
        # broken pipe only surfaces on a write. Stopping tracking stops
        # every stream, so this loop would find nothing to send, never
        # yield, and spin forever -- one immortal thread per stop/start,
        # each burning CPU and a request slot until the server was
        # restarted. A keepalive gives the socket something to fail on,
        # and an idle cutoff ends the response when the session is over.
        idle_since = time.time()
        last_ping = time.time()
        while True:
            sent = False
            for n in list(seqs):
                if min_dt and time.time() - last_sent[n] < min_dt:
                    continue          # per-node view-fps throttle
                st = streams.get(n)
                if st is None or not st.started:
                    continue
                frame, seq, _ts = st.get_frame(seqs[n], timeout=0.02)
                if frame is None or seq == seqs[n]:
                    continue
                seqs[n] = seq
                last_sent[n] = time.time()
                h, w = frame.shape[:2]
                ow, oh = (maxw, round(h * maxw / w)) if w > maxw else (w, h)
                if (ow, oh) != (w, h):
                    frame = cv2.resize(frame, (ow, oh))
                m = _view_maps(n, w, h, ow, oh, maps) if undistort else None
                if m is not None:
                    frame = cv2.remap(frame, *m, cv2.INTER_LINEAR)
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                b = jpg.tobytes()
                yield f"{n},{len(b)}\n".encode() + b
                sent = True
            now = time.time()
            if sent:
                idle_since = now
            else:
                if now - last_ping > 2.0:
                    # zero-length record on a node the client does not
                    # know: the parser skips it, but writing it raises if
                    # the browser has gone
                    last_ping = now
                    yield b"-1,0\n"
                if now - idle_since > 10.0:
                    return          # nothing has streamed for 10 s: done
                time.sleep(0.05)

    return Response(gen(), mimetype="application/octet-stream")


# ---------------------------------------------------------------- tracking

def _start_stream(node, width, height, fps):
    st = streams.get(node)
    if (st is not None and st.started
            and getattr(st, "_settings", None) == (width, height, fps)):
        return dict(st.info)          # already running as requested: reuse
    cam = next((c for c in cameras.list_cameras() if c["node"] == node), None)
    if cam is None:
        return None
    if st is not None:
        st.stop()
    else:
        st = streams[node] = cameras.CameraStream()
    try:
        return st.start(cam, width, height, fps)
    except RuntimeError:
        streams.pop(node, None)
        return None


@app.post("/api/host/track/start")
def api_track_start():
    """(Re)start tag tracking: opens the requested camera streams at the
    view fps and launches the detection loop after a warm-up delay."""
    body = request.get_json(force=True)
    cam_list = body.get("cameras") or []
    view_fps = float(body.get("view_fps") or 10)
    keep = set(int(n) for n in (body.get("keep") or []))
    wanted = set(int(c["node"]) for c in cam_list)
    tracker.stop()
    started, tr_cams = [], {}
    with stream_lock:
        # release cameras that are no longer tracked (USB stays honest)
        for n in list(streams.keys()):
            if n not in wanted and n not in keep:
                streams.pop(n).stop()
        for cc in cam_list:
            node = int(cc["node"])
            info = _start_stream(node, int(cc.get("width", 1280)),
                                 int(cc.get("height", 720)), view_fps)
            if info is None:
                continue
            started.append(node)
            entry = {"K": None, "dist": None, "cal_size": None}
            slug = cc.get("cal_slug")
            if slug and storage.valid_slug(slug):
                try:
                    data = get_store().get(slug)
                    i = (data or {}).get("intrinsic") or {}
                    if i.get("camera_matrix"):
                        entry = {"K": i["camera_matrix"],
                                 "dist": i.get("dist_coeffs"),
                                 "cal_size": i.get("image_size")}
                except Exception:
                    pass
            tr_cams[node] = entry
            _set_view_cal(node, entry)
    tracker.start(tr_cams, body.get("track_fps") or 5,
                  body.get("marker_mm") or 40,
                  warmup_s=float(body.get("warmup_s", 3.0)))
    return jsonify({"ok": True, "started": started})


@app.post("/api/host/track/config")
def api_track_config():
    body = request.get_json(force=True)
    tracker.configure(track_fps=body.get("track_fps"),
                      marker_mm=body.get("marker_mm"))
    return jsonify({"ok": True})


@app.post("/api/host/track/stop")
def api_track_stop():
    body = request.get_json(force=True) if request.data else {}
    hcal.stop()
    tracker.stop()
    with stream_lock:
        for n in body.get("stop_streams") or []:
            st = streams.pop(int(n), None)
            if st is not None:
                st.stop()
    return jsonify({"ok": True})


@app.post("/api/host/track/world")
def api_track_world():
    body = request.get_json(force=True) if request.data else {}
    try:
        root = body.get("root")
        return jsonify(tracker.world_solve(
            None if root in (None, "", "auto") else int(root),
            body.get("marker_mm"), body.get("world_ref")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ------------------------------------------------------------- styluses
# A stylus is a physical object (tag layout + tip), kept with the host's
# rig settings: {id: {id, name, tag_mm, tip_diameter_mm, calibration}}.
scal = stylus_mod.StylusCalSession(tracker)


@app.get("/api/host/styluses")
def api_styluses():
    return jsonify(list(load_config().get("styluses", {}).values()))


@app.put("/api/host/styluses/<sid>")
def api_stylus_put(sid):
    body = request.get_json(force=True) or {}
    if not isinstance(body, dict) or not storage.valid_slug(sid):
        return jsonify({"ok": False, "error": "bad stylus"}), 400
    body["id"] = sid
    cfg = load_config()
    cfg.setdefault("styluses", {})[sid] = body
    save_config(cfg)
    return jsonify({"ok": True})


@app.delete("/api/host/styluses/<sid>")
def api_stylus_delete(sid):
    cfg = load_config()
    cfg.setdefault("styluses", {}).pop(sid, None)
    save_config(cfg)
    return jsonify({"ok": True})


@app.post("/api/host/stylus/cal/start")
def api_scal_start():
    if not tracker.cams:
        return jsonify({"ok": False, "error": "camera streams are not running"}), 400
    scal.start()
    return jsonify({"ok": True})


@app.post("/api/host/stylus/cal/snap")
def api_scal_snap():
    body = request.get_json(force=True) if request.data else {}
    return jsonify(scal.snap(body.get("tag_mm") or 20))


@app.post("/api/host/stylus/cal/solve")
def api_scal_solve():
    """body: {poses: {node: T_world_cam}, tag_mm}."""
    body = request.get_json(force=True) if request.data else {}
    import numpy as np
    cam_T = {int(n): np.asarray(T, float) for n, T in (body.get("poses") or {}).items()}
    if len(cam_T) < 2:
        return jsonify({"ok": False, "error": "needs world poses for 2+ cameras "
                        "— run World Calibration first"})
    try:
        return jsonify(scal.solve(cam_T, body.get("tag_mm") or 20))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/host/track/hcal/start")
def api_hcal_start():
    """Start collecting hand samples on the cameras the pose tab is tracking
    (their intrinsics were loaded by track/start). body: {rot: {node: deg}}"""
    body = request.get_json(force=True) if request.data else {}
    if not tracker.cams:
        return jsonify({"ok": False, "error": "start the pose tab's tracking first"}), 400
    rot = {int(n): r for n, r in (body.get("rot") or {}).items()}
    hcal.start({n: rot.get(n, 0) for n in tracker.cams})
    return jsonify({"ok": True, "cameras": sorted(tracker.cams)})


@app.get("/api/host/track/hcal/status")
def api_hcal_status():
    return jsonify(hcal.status())


@app.post("/api/host/track/hcal/solve")
def api_hcal_solve():
    """Stop collecting and solve. body: {world_ref, compare: {node: T},
    keep_scale, marker_mm}."""
    body = request.get_json(force=True) if request.data else {}
    try:
        return jsonify(hcal.solve(
            world_ref=body.get("world_ref"), compare=body.get("compare"),
            keep_scale=bool(body.get("keep_scale")),
            frame_from_camera=tracker_mod.Tracker._frame_from_camera,
            tag_mm=body.get("marker_mm")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/host/track/hcal/cancel")
def api_hcal_cancel():
    hcal.stop()
    return jsonify({"ok": True})


@app.post("/api/host/track/wcal/start")
def api_wcal_start():
    tracker.wcal_start()
    return jsonify({"ok": True})


@app.post("/api/host/track/wcal/snap")
def api_wcal_snap():
    body = request.get_json(force=True) if request.data else {}
    return jsonify(tracker.wcal_snap(body.get("marker_mm")))


@app.post("/api/host/track/wcal/solve")
def api_wcal_solve():
    body = request.get_json(force=True) if request.data else {}
    try:
        root = body.get("root")
        return jsonify(tracker.wcal_solve(
            None if root in (None, "", "auto") else int(root),
            body.get("marker_mm"), body.get("world_ref")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/host/rig")
def api_rig_get():
    """Per-machine rig settings: which labelled calibration is on which USB
    port, per-camera capture resolution, which cameras are enabled.

    These describe the physical rig, so they belong to the host and not to
    whichever browser happens to be driving it. Kept in the browser they
    are lost the moment you connect from a different machine -- and the
    fallback is the alphabetically first calibration for every camera,
    which silently assigns them all to the same one.
    """
    return jsonify(load_config().get("rig", {}))


@app.put("/api/host/rig")
def api_rig_put():
    """Merge the posted keys into the stored rig settings."""
    body = request.get_json(force=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "expected an object"}), 400
    cfg = load_config()
    rig = cfg.setdefault("rig", {})
    for k, v in body.items():
        if v is None:
            rig.pop(k, None)
        else:
            rig[k] = v
    save_config(cfg)
    return jsonify(rig)


# ------------------------------------------------------------ live tracking

@app.get("/api/host/detectors")
def api_detectors():
    """What can be tracked here, and why anything unavailable isn't."""
    return jsonify({"detectors": detectors_mod.registry(),
                    "gpu": live.gpu.sample()})


@app.post("/api/host/live/start")
def api_live_start():
    """Open the requested cameras and start fusing detections into world
    space. Cameras without a world pose are refused up front rather than
    silently producing nothing."""
    body = request.get_json(force=True)
    cam_list = body.get("cameras") or []
    keys = body.get("detectors") or ["aruco"]
    view_fps = float(body.get("view_fps") or 10)
    missing = [int(c["node"]) for c in cam_list if not c.get("T_world_cam")]
    if missing:
        return jsonify({
            "ok": False, "missing_extrinsics": missing,
            "error": ("no saved world pose for " +
                      ", ".join("video%d" % n for n in missing) +
                      " — run Camera Pose Estimation for those cameras, or "
                      "uncheck them")}), 400
    if not cam_list:
        return jsonify({"ok": False, "error": "no cameras selected"}), 400
    live.stop()
    cams = {}
    with stream_lock:
        for cc in cam_list:
            node = int(cc["node"])
            info = _start_stream(node, int(cc.get("width", 1280)),
                                 int(cc.get("height", 720)), view_fps)
            if info is None:
                continue
            cams[node] = {"K": cc.get("K"), "dist": cc.get("dist"),
                          "cal_size": cc.get("cal_size"),
                          "T_world_cam": cc.get("T_world_cam"),
                          "rot": cc.get("rot")}
            _set_view_cal(node, cams[node])
    res = live.start(cams, keys,
                     track_fps=float(body.get("track_fps") or 10),
                     warmup_s=float(body.get("warmup_s", 1.5)),
                     marker_mm=float(body.get("marker_mm") or 40),
                     # two hands per person; more invites duplicates
                     max_hands=int(body.get("max_hands") or 2),
                     stylus=body.get("stylus"))
    res["started"] = sorted(cams)
    res["failed"] = sorted(set(int(c["node"]) for c in cam_list) - set(cams))
    return jsonify(res)


@app.get("/api/host/live/hand_debug")
def api_live_hand_debug():
    """Recent hand-grouping evidence: per tick, every camera's detection,
    the gap between each pair of palm rays, and how they were grouped."""
    return jsonify(list(live.hand_debug))


# --------------------------------------------------------- surface mapping
@app.post("/api/host/map/config")
def api_map_config():
    smap.configure(**(request.get_json(force=True) or {}))
    return jsonify({"ok": True, "cfg": smap.cfg})


@app.post("/api/host/map/arm")
def api_map_arm():
    smap.arm((request.get_json(force=True) or {}).get("on"))
    return jsonify({"ok": True})


@app.post("/api/host/map/tap")
def api_map_tap():
    smap.tap()
    return jsonify({"ok": True})


@app.post("/api/host/map/undo")
def api_map_undo():
    smap.undo()
    return jsonify({"ok": True})


@app.post("/api/host/map/clear")
def api_map_clear():
    smap.clear()
    return jsonify({"ok": True})


@app.get("/api/host/map/points")
def api_map_points():
    return jsonify(smap.snapshot(int(request.args.get("since", 0))))


@app.post("/api/host/map/compute")
def api_map_compute():
    body = request.get_json(force=True) if request.data else {}
    try:
        return jsonify(smap.compute(k=int(body.get("k") or 10),
                                    max_edge_factor=float(body.get("max_edge") or 3.0)))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/host/map/export")
def api_map_export():
    try:
        name, mime, data = smap.export(request.args.get("fmt", "ply"),
                                       relative=request.args.get("rel", "1") == "1")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return Response(data, mimetype=mime,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/host/tagsizes")
def api_tagsizes():
    return jsonify(tag_sizes.listing())


@app.delete("/api/host/tagsizes/<int:tid>")
def api_tagsize_forget(tid):
    tag_sizes.forget(tid)
    return jsonify({"ok": True})


@app.post("/api/host/live/stop")
def api_live_stop():
    live.stop()
    tag_sizes.flush()
    return jsonify({"ok": True})


@app.get("/api/host/live/results")
def api_live_results():
    snap = live.snapshot()
    snap["metrics"] = tracker.metrics.sample()
    snap["capture"] = {
        str(n): {"fps": st.fps_actual, "target": st.info.get("fps"),
                 "width": st.info.get("width"),
                 "height": st.info.get("height"),
                 "dead": not st.started, "error": st.error}
        for n, st in streams.items() if st.started or st.error}
    return jsonify(snap)


@app.post("/api/host/track/repose")
def api_track_repose():
    """Rebuild the world frame around a chosen near-top-down camera."""
    body = request.get_json(force=True)
    ref = body.get("reference_node")
    if ref is None:
        return jsonify({"ok": False, "error": "no reference camera"}), 400
    return jsonify(tracker_mod.Tracker.repose_world(
        body.get("poses") or {}, ref,
        yaw_quadrant=int(body.get("yaw_quadrant") or 0),
        mode=body.get("mode") or "topdown"))


@app.post("/api/host/track/verify")
def api_track_verify():
    """Check saved camera extrinsics against a live view of the test block.

    Poses come from the client (which already holds the calibration files)
    so the server stays stateless about extrinsics.
    """
    body = request.get_json(force=True)
    return jsonify(tracker.verify_poses(
        body.get("poses") or {},
        marker_mm=body.get("marker_mm"),
        tol_px=float(body.get("tol_px") or 3.0),
        tol_mm=float(body.get("tol_mm") or 8.0)))


@app.get("/api/host/track/results")
def api_track_results():
    snap = tracker.snapshot()
    snap["metrics"] = tracker.metrics.sample()
    # dead streams are reported too (with their error) — a camera that opened
    # and then failed is exactly the case the client needs to explain
    snap["capture"] = {
        str(n): {"fps": st.fps_actual,
                 "target": st.info.get("fps"),
                 "width": st.info.get("width"),
                 "height": st.info.get("height"),
                 "dead": not st.started,
                 "error": st.error}
        for n, st in streams.items() if st.started or st.error}
    return jsonify(snap)


# ------------------------------------------------------- calibration storage

@app.get("/api/host/storage")
def api_storage_status():
    try:
        st = get_store().status()
    except Exception as e:
        st = {"ok": False, "error": str(e)}
    st["config"] = load_config().get("storage", {})
    return jsonify(st)


@app.put("/api/host/storage")
def api_storage_config():
    body = request.get_json(force=True)
    cfg = load_config()
    allowed = {k: v for k, v in body.items()
               if k in ("type", "dir_path", "env_file") and v}
    if allowed.get("type") not in ("dir", "s3"):
        return jsonify({"error": "type must be 'dir' or 's3'"}), 400
    cfg["storage"] = allowed
    save_config(cfg)
    try:
        st = get_store(rebuild=True).status()
    except Exception as e:
        st = {"ok": False, "error": str(e)}
    st["config"] = allowed
    return jsonify(st)


@app.get("/api/host/storage/reveal")
def api_storage_reveal():
    store = get_store()
    if not hasattr(store, "reveal"):
        return jsonify({"error": "reveal only applies to S3 storage"}), 400
    return jsonify(store.reveal())


@app.get("/api/host/storage/alt")
def api_storage_alt():
    """Status + contents of the non-selected backend, for the copy-across
    button (e.g. active=S3 -> alt is the local directory)."""
    try:
        store = get_alt_store()
        st = store.status()
        if st.get("ok"):
            st["slugs"] = [e["slug"] for e in store.list()]
    except Exception as e:
        st = {"ok": False, "error": str(e)}
    st["config"] = alt_storage_cfg()
    return jsonify(st)


@app.post("/api/host/calibrations/<slug>/copy_alt")
def api_cal_copy_alt(slug):
    """Copy one calibration from the active backend to the other one."""
    if not storage.valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    try:
        data = get_store().get(slug)
        if data is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(get_alt_store().put(slug, data))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.get("/api/host/calibrations")
def api_cal_list():
    try:
        return jsonify(get_store().list())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.get("/api/host/calibrations/<slug>")
def api_cal_get(slug):
    if not storage.valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    try:
        data = get_store().get(slug)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@app.put("/api/host/calibrations/<slug>")
def api_cal_put(slug):
    if not storage.valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    data = request.get_json(force=True)
    if not isinstance(data, dict) or "intrinsic" not in data:
        return jsonify({"error": "not a calibration file"}), 400
    try:
        return jsonify(get_store().put(slug, data))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.delete("/api/host/calibrations/<slug>")
def api_cal_delete(slug):
    if not storage.valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    try:
        r = get_store().delete(slug)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    if r.get("error"):
        return jsonify(r), 404
    return jsonify(r)


@app.post("/api/host/calibrations/<slug>/rename")
def api_cal_rename(slug):
    body = request.get_json(force=True)
    new_slug = body.get("new_slug", "")
    if not (storage.valid_slug(slug) and storage.valid_slug(new_slug)):
        return jsonify({"error": "bad slug"}), 400
    store = get_store()
    r = store.rename(slug, new_slug)
    if r.get("error"):
        return jsonify(r), 409
    # keep the file's own identity fields in sync with its new key
    data = store.get(new_slug)
    if isinstance(data, dict):
        data["slug"] = new_slug
        if "label" in body:
            data.setdefault("camera", {})["assigned_label"] = body["label"] or None
            if body["label"]:
                data["name"] = body["label"]
        store.put(new_slug, data)
    return jsonify({"ok": True, "slug": new_slug})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--storage", default=None,
                    help="'dir:/path' or 's3' — overrides saved config")
    ap.add_argument("--env-file", default=None,
                    help=".env with AWS creds for --storage s3 "
                         "(process env vars take precedence)")
    args = ap.parse_args()
    if args.storage:
        cfg = load_config()
        if args.storage.startswith("dir:"):
            cfg["storage"] = {"type": "dir", "dir_path": args.storage[4:]}
        elif args.storage == "s3":
            cfg["storage"] = {"type": "s3"}
            if args.env_file:
                cfg["storage"]["env_file"] = args.env_file
        else:
            ap.error("--storage must be 'dir:/path' or 's3'")
        save_config(cfg)
    print("storage:", get_store().status())
    print(f"local bridge on http://{args.bind}:{args.port}  "
          f"({len(cameras.list_cameras())} camera(s) detected)")
    app.run(host=args.bind, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
