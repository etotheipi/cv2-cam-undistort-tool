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
    from . import (cameras, detectors as detectors_mod, livetrack,
                   storage, tracker as tracker_mod)
except ImportError:          # running as a plain script
    import cameras
    import detectors as detectors_mod
    import livetrack
    import storage
    import tracker as tracker_mod

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

app = Flask(__name__)
streams = {}                 # node -> CameraStream (several cameras at once)
stream_lock = threading.Lock()
tracker = tracker_mod.Tracker(streams)
live = livetrack.LiveTracker(streams)

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

    def gen():
        seqs = {n: 0 for n in nodes}
        last_sent = {n: 0.0 for n in nodes}
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
                if w > maxw:
                    frame = cv2.resize(frame, (maxw, round(h * maxw / w)))
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                b = jpg.tobytes()
                yield f"{n},{len(b)}\n".encode() + b
                sent = True
            if not sent:
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
                          "T_world_cam": cc.get("T_world_cam")}
    res = live.start(cams, keys,
                     track_fps=float(body.get("track_fps") or 10),
                     warmup_s=float(body.get("warmup_s", 1.5)),
                     marker_mm=float(body.get("marker_mm") or 40),
                     max_hands=int(body.get("max_hands") or 4))
    res["started"] = sorted(cams)
    res["failed"] = sorted(set(int(c["node"]) for c in cam_list) - set(cams))
    return jsonify(res)


@app.post("/api/host/live/stop")
def api_live_stop():
    live.stop()
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
