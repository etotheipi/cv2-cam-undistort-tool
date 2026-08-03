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
        poses, then a pose-graph solve over the camera<->tag graph.

        Anchoring: the gauge is fixed on the MOST-OBSERVED tag inside the
        root tag's connected component (best-constrained node), and the
        finished solution is re-expressed in the root tag's frame — the
        root still defines world coordinates and must be visible.

        Refinement: block least squares alternating (a) one multi-tag
        solvePnP per camera over the corners of every world-known tag it
        sees (true reprojection LS, LM-refined from the current estimate)
        and (b) per-tag weighted pose averaging across cameras, weighted
        by the observed pixel area — which naturally down-weights distant
        and oblique views (area ~ cos(tilt)/distance^2)."""
        root_id = int(root_id)
        marker = float(marker_mm or self.marker_mm)
        half = marker / 2.0
        objp = np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], np.float32)
        obs, views, k_of = {}, {}, {}
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
            k_of[node] = (K, dist)
            tag2d, t_obs = [], {}
            if ids is not None:
                for c4, tid in zip(corners, ids.ravel()):
                    pts = c4.reshape(4, 2).astype(np.float32)
                    entry = {"id": int(tid),
                             "corners": np.round(pts, 1).tolist()}
                    try:
                        ok, rvec, tvec = cv2.solvePnP(
                            objp, pts, K, dist,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if ok:
                            R, _ = cv2.Rodrigues(rvec)
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = tvec.ravel()
                            # shoelace area of the observed quad, px^2
                            x, y = pts[:, 0], pts[:, 1]
                            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                                             np.dot(y, np.roll(x, -1)))
                            t_obs[int(tid)] = {"T": T, "pts": pts,
                                               "area": float(max(area, 1.0))}
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
            if t_obs:
                obs[node] = t_obs
        if not any(root_id in t for t in obs.values()):
            return {"ok": False, "views": views,
                    "error": f"Tag {root_id} is not visible to any camera"}
        # connected component containing the root (bipartite BFS)
        comp_tags, comp_cams = {root_id}, set()
        changed = True
        while changed:
            changed = False
            for node, tags in obs.items():
                if node not in comp_cams and any(t in comp_tags for t in tags):
                    comp_cams.add(node)
                    changed = True
            for node in comp_cams:
                for tid in obs[node]:
                    if tid not in comp_tags:
                        comp_tags.add(tid)
                        changed = True
        # anchor: most observations; ties prefer the root, then lowest id
        counts = {t: sum(1 for n in comp_cams if t in obs[n])
                  for t in comp_tags}
        anchor = max(comp_tags,
                     key=lambda t: (counts[t], t == root_id, -t))
        # BFS initialization from the anchor
        tags_T = {anchor: np.eye(4)}
        cams_T = {}
        changed = True
        while changed:
            changed = False
            for node in comp_cams:
                if node in cams_T:
                    continue
                for tid, o in obs[node].items():
                    if tid in tags_T:
                        cams_T[node] = tags_T[tid] @ self._inv(o["T"])
                        changed = True
                        break
            for node in comp_cams:
                Tc = cams_T.get(node)
                if Tc is None:
                    continue
                for tid, o in obs[node].items():
                    if tid not in tags_T:
                        tags_T[tid] = Tc @ o["T"]
                        changed = True
        corners_h = np.array([[-half, half, 0, 1], [half, half, 0, 1],
                              [half, -half, 0, 1], [-half, -half, 0, 1]],
                             np.float64).T
        # block least-squares refinement
        for _it in range(12):
            for node in comp_cams:
                if node not in cams_T:
                    continue
                p3, p2 = [], []
                for tid, o in obs[node].items():
                    if tid in tags_T:
                        p3.append((tags_T[tid] @ corners_h).T[:, :3])
                        p2.append(o["pts"])
                if not p3:
                    continue
                P3 = np.concatenate(p3).astype(np.float32)
                P2 = np.concatenate(p2).astype(np.float32).reshape(-1, 1, 2)
                K, dist = k_of[node]
                Tcw = self._inv(cams_T[node])
                rvec, _ = cv2.Rodrigues(Tcw[:3, :3])
                tvec = Tcw[:3, 3].reshape(3, 1).copy()
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        P3, P2, K, dist, rvec, tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                except cv2.error:
                    continue
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    Tcw = np.eye(4)
                    Tcw[:3, :3] = R
                    Tcw[:3, 3] = tvec.ravel()
                    cams_T[node] = self._inv(Tcw)
            for tid in comp_tags:
                if tid == anchor:
                    continue
                Ms, ts, ws = [], [], []
                for node in comp_cams:
                    o = obs[node].get(tid)
                    if o is None or node not in cams_T:
                        continue
                    Tw = cams_T[node] @ o["T"]
                    w = o["area"]
                    Ms.append(w * Tw[:3, :3])
                    ts.append(w * Tw[:3, 3])
                    ws.append(w)
                if not ws:
                    continue
                U, _s, Vt = np.linalg.svd(np.sum(Ms, axis=0))
                R = U @ np.diag(
                    [1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = np.sum(ts, axis=0) / sum(ws)
                tags_T[tid] = T
        # residual: reprojection RMS over every observation in the solve
        errs = []
        for node in comp_cams:
            if node not in cams_T:
                continue
            K, dist = k_of[node]
            Tcw = self._inv(cams_T[node])
            rvec, _ = cv2.Rodrigues(Tcw[:3, :3])
            tvec = Tcw[:3, 3]
            for tid, o in obs[node].items():
                if tid in tags_T:
                    wc = (tags_T[tid] @ corners_h).T[:, :3]
                    proj, _ = cv2.projectPoints(wc, rvec, tvec, K, dist)
                    errs.append(np.linalg.norm(
                        proj.reshape(-1, 2) - o["pts"], axis=1))
        rms = (round(float(np.sqrt(np.mean(np.concatenate(errs) ** 2))), 3)
               if errs else None)
        # gauge shift: express everything in the ROOT tag's frame
        shift = self._inv(tags_T[root_id])
        tags_T = {t: shift @ T for t, T in tags_T.items()}
        cams_T = {n: shift @ T for n, T in cams_T.items()}
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
                "anchor": int(anchor), "rms_px": rms,
                "cameras": cams_out, "tags": tags_out, "views": views,
                "unlinked_cameras": sorted(set(views) - set(cams_T)),
                "unlinked_tags": sorted(seen_ids - set(tags_T))}
