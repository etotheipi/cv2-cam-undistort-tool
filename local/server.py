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
    from . import cameras, storage
except ImportError:          # running as a plain script
    import cameras
    import storage

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

app = Flask(__name__)
streams = {}                 # node -> CameraStream (several cameras at once)
stream_lock = threading.Lock()

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
        frame, _ = st.get_frame(0, timeout=3.0)
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
        frame, seq = st.get_frame(seq, timeout=2.0)
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

    def gen():
        seqs = {n: 0 for n in nodes}
        while True:
            sent = False
            for n in list(seqs):
                st = streams.get(n)
                if st is None or not st.started:
                    continue
                frame, seq = st.get_frame(seqs[n], timeout=0.02)
                if frame is None or seq == seqs[n]:
                    continue
                seqs[n] = seq
                h, w = frame.shape[:2]
                if w > maxw:
                    frame = cv2.resize(frame, (maxw, round(h * maxw / w)))
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    continue
                b = jpg.tobytes()
                yield f"{n},{len(b)}\n".encode() + b
                sent = True
            if not sent:
                time.sleep(0.05)

    return Response(gen(), mimetype="application/octet-stream")


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
