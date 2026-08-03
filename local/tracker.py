"""Server-side ArUco (AprilTag 36h11) tag tracker for the Tracking tab.

Detection runs natively here rather than in the browser's Pyodide: it is
multi-core, ~5x faster, and reflects what an eventual deployment (e.g. a
Raspberry Pi tracker) actually runs — so the load metrics this module
reports are meaningful for capacity planning.

One tracker thread sweeps every tracked camera at the configured rate,
detecting on each camera's newest frame (frames between ticks are simply
skipped). Per-tag distance comes from solvePnP(IPPE_SQUARE) on the four
corners with the camera's calibration; uncalibrated cameras fall back to
a rough focal guess (f = 0.8*width, ~64 deg HFOV) and are flagged approx.
"""

import base64
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


class Metrics:
    """Process/system usage from /proc, sampled between calls."""

    def __init__(self):
        self.ncpu = os.cpu_count() or 1
        self.clk = os.sysconf("SC_CLK_TCK")
        self._last = None
        self._cpu = {}          # last computed CPU figures (1s-averaged)

    def _read(self):
        with open("/proc/self/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        proc = (int(parts[11]) + int(parts[12])) / self.clk   # utime+stime
        with open("/proc/stat") as f:
            cpu = f.readline().split()[1:]
        total = sum(int(x) for x in cpu)
        idle = int(cpu[3]) + int(cpu[4])
        rss = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        return {"t": time.time(), "proc": proc, "total": total,
                "idle": idle, "rss": rss}

    def sample(self):
        cur = self._read()
        out = {"rss_mb": round(cur["rss"] / 1048576, 1),
               "ncpu": self.ncpu,
               "loadavg": [round(v, 2) for v in os.getloadavg()]}
        try:
            with open("/proc/meminfo") as f:
                memtotal = int(f.readline().split()[1]) * 1024
            out["rss_pct"] = round(100.0 * cur["rss"] / memtotal, 1)
        except (OSError, ValueError):
            pass
        # CPU figures averaged over >=1s windows so frequent polls don't
        # produce jittery (or missing) numbers
        if self._last is None:
            self._last = cur
        elif cur["t"] - self._last["t"] >= 1.0:
            last = self._last
            dt = cur["t"] - last["t"]
            self._cpu = {"proc_cpu_pct_one_core": round(
                100.0 * (cur["proc"] - last["proc"]) / dt, 1)}
            self._cpu["proc_cpu_pct_machine"] = round(
                self._cpu["proc_cpu_pct_one_core"] / self.ncpu, 1)
            dtotal = cur["total"] - last["total"]
            didle = cur["idle"] - last["idle"]
            if dtotal > 0:
                self._cpu["system_cpu_pct"] = round(
                    100.0 * (1 - didle / dtotal), 1)
            self._last = cur
        out.update(self._cpu)
        return out


class Tracker:
    def __init__(self, streams):
        self.streams = streams          # server's node -> CameraStream dict
        self.detector = cv2.aruco.ArucoDetector(DICTIONARY)
        self.metrics = Metrics()
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self.track_fps = 5.0
        self.marker_mm = 40.0
        self.warmup_s = 3.0
        self.cams = {}                  # node -> {"K","dist","cal_size"}
        self.results = {}               # node -> latest detection payload
        self.stats = {"achieved_fps": 0.0, "duty_pct": 0.0}

    def configure(self, track_fps=None, marker_mm=None):
        with self._lock:
            if track_fps:
                self.track_fps = float(track_fps)
            if marker_mm:
                self.marker_mm = float(marker_mm)

    def start(self, cams, track_fps, marker_mm, warmup_s=3.0):
        self.stop()
        self.cams = cams
        self.track_fps = float(track_fps)
        self.marker_mm = float(marker_mm)
        self.warmup_s = float(warmup_s)
        self.results = {}
        self.stats = {"achieved_fps": 0.0, "duty_pct": 0.0}
        if not cams:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self):
        return self._running

    def _K_for(self, node, w, h):
        c = self.cams.get(node) or {}
        if c.get("K"):
            cw, ch = c.get("cal_size") or (w, h)
            K = np.array(c["K"], np.float64)
            if (cw, ch) != (w, h):
                K = np.diag([w / cw, h / ch, 1.0]) @ K
            return K, np.array(c.get("dist") or [0] * 5, np.float64), False
        f = 0.8 * w                    # rough default for uncalibrated cams
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        return K, np.zeros(5), True

    def _detect(self, node, frame, marker_mm):
        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rej = self.detector.detectMarkers(gray)
        h, w = gray.shape
        tags = []
        if ids is not None and len(ids):
            K, dist, approx = self._K_for(node, w, h)
            half = marker_mm / 2.0
            # order matches aruco corners: TL, TR, BR, BL (y up on the tag)
            objp = np.array([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], np.float32)
            for c4, tid in zip(corners, ids.ravel()):
                pts = c4.reshape(4, 2)
                entry = {"id": int(tid),
                         "corners": np.round(pts, 1).tolist()}
                try:
                    ok, _rvec, tvec = cv2.solvePnP(
                        objp, pts.astype(np.float32), K, dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        entry["distance_mm"] = round(
                            float(np.linalg.norm(tvec)), 1)
                        entry["approx"] = bool(approx)
                except cv2.error:
                    pass
                tags.append(entry)
        return {"ts": round(time.time(), 3), "n": len(tags), "tags": tags,
                "size": [w, h],
                "detect_ms": round((time.time() - t0) * 1000, 2)}

    def _loop(self):
        time.sleep(self.warmup_s)      # camera startup + auto-exposure
        seqs = {}
        ticks = deque(maxlen=20)
        busys = deque(maxlen=20)
        while self._running:
            t0 = time.time()
            with self._lock:
                fps = self.track_fps
                marker = self.marker_mm
            for node in list(self.cams.keys()):
                if not self._running:
                    return
                st = self.streams.get(node)
                if st is None or not st.started:
                    continue
                frame, seq = st.get_frame(seqs.get(node, 0), timeout=0.005)
                if frame is None or seq == seqs.get(node):
                    continue
                seqs[node] = seq
                self.results[node] = self._detect(node, frame, marker)
            busy = time.time() - t0
            ticks.append(t0)
            busys.append(busy)
            if len(ticks) >= 2:
                span = ticks[-1] - ticks[0] + busy
                if span > 0:
                    self.stats["achieved_fps"] = round(
                        (len(ticks) - 1) / (ticks[-1] - ticks[0]), 2) \
                        if ticks[-1] > ticks[0] else 0.0
                    self.stats["duty_pct"] = round(
                        100.0 * sum(busys) / span, 1)
            time.sleep(max(0.0, 1.0 / max(fps, 0.1) - busy))

    def snapshot(self):
        return {"running": self._running,
                "stats": dict(self.stats),
                "config": {"track_fps": self.track_fps,
                           "marker_mm": self.marker_mm},
                "results": dict(self.results)}

    # ------------------------------------------------- world pose graph
    @staticmethod
    def _inv(T):
        R, t = T[:3, :3], T[:3, 3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    def world_solve(self, root_id, marker_mm=None):
        """One-shot: latest frame from every tracked camera, per-tag PnP
        poses, then a BFS over the camera<->tag graph rooted at the world
        tag. Observation obs[cam][tag] = T_cam<-tag; a camera seeing a
        world-known tag gets T_w<-cam = T_w<-tag @ inv(T_cam<-tag), and its
        other tags join via T_w<-tag = T_w<-cam @ T_cam<-tag. Nodes with no
        path to the root are reported as omitted."""
        root_id = int(root_id)
        marker = float(marker_mm or self.marker_mm)
        half = marker / 2.0
        objp = np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], np.float32)
        obs, views = {}, {}
        for node in list(self.cams.keys()):
            st = self.streams.get(node)
            if st is None or not st.started:
                continue
            frame, _ = st.get_frame(0, timeout=0.5)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rej = self.detector.detectMarkers(gray)
            h, w = gray.shape
            K, dist, approx = self._K_for(node, w, h)
            tag2d, t_by_tag = [], {}
            if ids is not None:
                for c4, tid in zip(corners, ids.ravel()):
                    pts = c4.reshape(4, 2)
                    entry = {"id": int(tid),
                             "corners": np.round(pts, 1).tolist()}
                    try:
                        ok, rvec, tvec = cv2.solvePnP(
                            objp, pts.astype(np.float32), K, dist,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if ok:
                            R, _ = cv2.Rodrigues(rvec)
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = tvec.ravel()
                            t_by_tag[int(tid)] = T
                            entry["distance_mm"] = round(
                                float(np.linalg.norm(tvec)), 1)
                            entry["approx"] = bool(approx)
                    except cv2.error:
                        pass
                    tag2d.append(entry)
            ok_enc, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
            views[node] = {
                "jpg_b64": base64.b64encode(jpg.tobytes()).decode()
                           if ok_enc else None,
                "size": [w, h], "tags": tag2d, "approx": bool(approx)}
            if t_by_tag:
                obs[node] = t_by_tag
        if not any(root_id in t for t in obs.values()):
            return {"ok": False, "views": views,
                    "error": f"Tag {root_id} is not visible to any camera"}
        tags_T = {root_id: np.eye(4)}
        cams_T = {}
        changed = True
        while changed:
            changed = False
            for node, tags in obs.items():
                if node in cams_T:
                    continue
                for tid, T in tags.items():
                    if tid in tags_T:
                        cams_T[node] = tags_T[tid] @ self._inv(T)
                        changed = True
                        break
            for node, tags in obs.items():
                Tc = cams_T.get(node)
                if Tc is None:
                    continue
                for tid, T in tags.items():
                    if tid not in tags_T:
                        tags_T[tid] = Tc @ T
                        changed = True
        corners_h = np.array([[-half, half, 0, 1], [half, half, 0, 1],
                              [half, -half, 0, 1], [-half, -half, 0, 1]]).T
        tags_out = [{"id": tid,
                     "corners_world": np.round((T @ corners_h).T[:, :3],
                                               1).tolist(),
                     "T": np.round(T, 5).tolist()}
                    for tid, T in tags_T.items()]
        cams_out = [{"node": node, "T": np.round(T, 5).tolist(),
                     "pos": np.round(T[:3, 3], 1).tolist(),
                     "seen": sorted(obs[node].keys())}
                    for node, T in cams_T.items()]
        seen_ids = set()
        for t in obs.values():
            seen_ids.update(t.keys())
        return {"ok": True, "root": root_id, "marker_mm": marker,
                "cameras": cams_out, "tags": tags_out, "views": views,
                "unlinked_cameras": sorted(set(views) - set(cams_T)),
                "unlinked_tags": sorted(seen_ids - set(tags_T))}
