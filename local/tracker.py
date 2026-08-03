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

    @staticmethod
    def _pnp_square(objp, pts, K, dist):
        """Robust single-tag pose. cv2 5.0's IPPE_SQUARE can return badly
        wrong poses on near-degenerate (small/axis-aligned) quads, so every
        candidate is verified by reprojection, SQPNP is the fallback, and
        the winner gets an LM polish. Returns (rvec, tvec, rms) or None."""
        best = None
        for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_SQPNP):
            try:
                ok, rvec, tvec = cv2.solvePnP(objp, pts, K, dist, flags=flag)
            except cv2.error:
                continue
            if not ok:
                continue
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
            if best is None or err < best[0]:
                best = (err, rvec, tvec)
            if err < 1.5:
                break
        if best is None:
            return None
        err, rvec, tvec = best
        try:
            rvec, tvec = cv2.solvePnPRefineLM(objp, pts, K, dist, rvec, tvec)
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
        except cv2.error:
            pass
        return rvec, tvec, err

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
                sol = self._pnp_square(objp, pts.astype(np.float32), K, dist)
                if sol is not None:
                    entry["distance_mm"] = round(
                        float(np.linalg.norm(sol[1])), 1)
                    entry["approx"] = bool(approx)
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

    def _capture_observations(self, marker):
        """Grab the newest frame from every tracked camera and detect tags
        with full poses. Returns (obs, views, k_of):
        obs[node][tid] = {"T": T_cam<-tag, "pts": 4x2, "area": px^2}."""
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
                    sol = self._pnp_square(objp, pts, K, dist)
                    if sol is not None:
                        rvec, tvec, _err = sol
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = tvec.ravel()
                        x, y = pts[:, 0], pts[:, 1]
                        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                                         np.dot(y, np.roll(x, -1)))
                        t_obs[int(tid)] = {"T": T, "pts": pts,
                                           "area": float(max(area, 1.0))}
                        entry["distance_mm"] = round(
                            float(np.linalg.norm(tvec)), 1)
                        entry["approx"] = bool(approx)
                    tag2d.append(entry)
            ok_enc, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
            views[node] = {
                "jpg_b64": base64.b64encode(jpg.tobytes()).decode()
                           if ok_enc else None,
                "size": [w, h], "tags": tag2d, "approx": bool(approx)}
            if t_obs:
                obs[node] = t_obs
        return obs, views, k_of

    def _solve_graph(self, snapshots, root_id, marker):
        """Joint pose-graph solve over N snapshots. Cameras have ONE pose
        shared by every snapshot; each (tag, snapshot) is its own free
        node (the reference block moves between snapshots). Camera nodes
        carry constraints across snapshots, so a camera that never sees
        the root still links in through any snapshot's shared tags.

        Residuals live in corner-pixel space: the camera update is one
        solvePnP over ALL corners it observed across ALL snapshots, which
        is exactly maximum likelihood under isotropic Gaussian corner
        noise — bearing information is tight, single-tag tilt is loose,
        with no hand-tuned per-pose weighting needed. Tag updates average
        across cameras weighted by observed pixel area (~cos(tilt)/d^2).

        Gauge: anchored on the most-observed (tag, snapshot) node in the
        root's component, then re-expressed in the frame of the root tag
        at its EARLIEST visible snapshot."""
        root_id = int(root_id)
        half = marker / 2.0
        # bipartite connectivity: cameras <-> (tag, snap)
        root_si = next((si for si, s in enumerate(snapshots)
                        if any(root_id in t for t in s["obs"].values())),
                       None)
        if root_si is None:
            return None
        root_key = (root_id, root_si)
        comp_tags, comp_cams = {root_key}, set()
        changed = True
        while changed:
            changed = False
            for si, s in enumerate(snapshots):
                for node, tags in s["obs"].items():
                    linked = any((tid, si) in comp_tags for tid in tags)
                    if node in comp_cams and not linked:
                        for tid in tags:
                            if (tid, si) not in comp_tags:
                                comp_tags.add((tid, si))
                                changed = True
                    elif linked:
                        if node not in comp_cams:
                            comp_cams.add(node)
                            changed = True
                        for tid in tags:
                            if (tid, si) not in comp_tags:
                                comp_tags.add((tid, si))
                                changed = True
        counts = {tk: sum(1 for n in comp_cams
                          if tk[0] in snapshots[tk[1]]["obs"].get(n, {}))
                  for tk in comp_tags}
        anchor = max(comp_tags,
                     key=lambda tk: (counts[tk], tk == root_key,
                                     -tk[0], -tk[1]))
        tags_T = {anchor: np.eye(4)}
        cams_T = {}
        changed = True
        while changed:
            changed = False
            for si, s in enumerate(snapshots):
                for node in comp_cams:
                    tags = s["obs"].get(node, {})
                    if node not in cams_T:
                        for tid, o in tags.items():
                            if (tid, si) in tags_T:
                                cams_T[node] = (tags_T[(tid, si)] @
                                                self._inv(o["T"]))
                                changed = True
                                break
                    Tc = cams_T.get(node)
                    if Tc is None:
                        continue
                    for tid, o in tags.items():
                        if (tid, si) in comp_tags and (tid, si) not in tags_T:
                            tags_T[(tid, si)] = Tc @ o["T"]
                            changed = True
        corners_h = np.array([[-half, half, 0, 1], [half, half, 0, 1],
                              [half, -half, 0, 1], [-half, -half, 0, 1]],
                             np.float64).T
        corners_tag = corners_h.T[:, :3]
        # ---- joint bundle adjustment (damped LM over all poses) ----
        # Block-alternation slides along the weakly-constrained tilt
        # directions of single-tag views; joint damped least squares over
        # (all camera poses + all non-anchor tag poses) with corner-pixel
        # residuals does not. Small problem => dense numeric Jacobian.
        cam_ids = sorted(n for n in comp_cams if n in cams_T)
        tag_ids = [tk for tk in sorted(tags_T) if tk != anchor]
        obs_list = []
        for si, s in enumerate(snapshots):
            for node, tags in s["obs"].items():
                if node not in cams_T or node not in s["k_of"]:
                    continue
                K, dist = s["k_of"][node]
                for tid, o in tags.items():
                    if (tid, si) in tags_T:
                        obs_list.append((node, (tid, si), o["pts"], K, dist))

        def pack():
            xs = []
            for n in cam_ids:
                Tcw = self._inv(cams_T[n])
                rv, _ = cv2.Rodrigues(Tcw[:3, :3])
                xs += list(rv.ravel()) + list(Tcw[:3, 3])
            for tkk in tag_ids:
                T = tags_T[tkk]
                rv, _ = cv2.Rodrigues(T[:3, :3])
                xs += list(rv.ravel()) + list(T[:3, 3])
            return np.array(xs, np.float64)

        def unpack(x):
            cams_cw, tags_w = {}, {anchor: np.eye(4)}
            i = 0
            for n in cam_ids:
                R, _ = cv2.Rodrigues(x[i:i + 3])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = x[i + 3:i + 6]
                cams_cw[n] = T                 # T_cam<-world
                i += 6
            for tkk in tag_ids:
                R, _ = cv2.Rodrigues(x[i:i + 3])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = x[i + 3:i + 6]
                tags_w[tkk] = T                # T_world<-tag
                i += 6
            return cams_cw, tags_w

        def residuals(x):
            cams_cw, tags_w = unpack(x)
            out = []
            for node, tkk, pts, K, dist in obs_list:
                Tw = tags_w[tkk]
                wc = (Tw[:3, :3] @ corners_tag.T).T + Tw[:3, 3]
                Tcw = cams_cw[node]
                rv, _ = cv2.Rodrigues(Tcw[:3, :3])
                proj, _ = cv2.projectPoints(wc, rv, Tcw[:3, 3], K, dist)
                out.append((proj.reshape(-1, 2) - pts).ravel())
            return np.concatenate(out) if out else np.zeros(0)

        x = pack()
        r = residuals(x)
        if len(r) and len(x):
            cost = float(r @ r)
            lam = 1e-3
            for _it in range(15):
                J = np.empty((len(r), len(x)))
                for j in range(len(x)):
                    eps = 1e-5 if (j % 6) < 3 else 1e-3
                    x2 = x.copy()
                    x2[j] += eps
                    J[:, j] = (residuals(x2) - r) / eps
                g = J.T @ r
                A = J.T @ J
                D = np.diag(np.diag(A) + 1e-9)
                improved = False
                dx = None
                for _try in range(6):
                    try:
                        dx = np.linalg.solve(A + lam * D, -g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    r2 = residuals(x + dx)
                    c2 = float(r2 @ r2)
                    if c2 < cost:
                        x = x + dx
                        r, cost = r2, c2
                        lam = max(lam / 3, 1e-7)
                        improved = True
                        break
                    lam *= 5
                if not improved:
                    break
                if dx is not None and float(np.linalg.norm(dx)) < 1e-5:
                    break
            cams_cw, tags_w = unpack(x)
            cams_T = {n: self._inv(T) for n, T in cams_cw.items()}
            tags_T = tags_w
        rms = (round(float(np.sqrt(np.mean(r ** 2))), 3)
               if len(r) else None)
        shift = self._inv(tags_T[root_key])
        tags_T = {tk: shift @ T for tk, T in tags_T.items()}
        cams_T = {n: shift @ T for n, T in cams_T.items()}
        tags_out = [{"id": tk[0], "snap": tk[1],
                     "corners_world": np.round((T @ corners_h).T[:, :3],
                                               1).tolist(),
                     "T": np.round(T, 5).tolist()}
                    for tk, T in tags_T.items()]
        cams_out = []
        for node, T in cams_T.items():
            seen = sorted({tid for s in snapshots
                           for tid in s["obs"].get(node, {})})
            cams_out.append({"node": node, "T": np.round(T, 5).tolist(),
                             "pos": np.round(T[:3, 3], 1).tolist(),
                             "seen": seen})
        all_cams, all_ids = set(), set()
        for s in snapshots:
            all_cams.update(s["views"].keys())
            for t in s["obs"].values():
                all_ids.update(t.keys())
        return {"ok": True, "root": root_id, "marker_mm": marker,
                "anchor": int(anchor[0]), "anchor_snap": int(anchor[1]),
                "rms_px": rms, "snap_count": len(snapshots),
                "cameras": cams_out, "tags": tags_out,
                "views_by_snap": [s["views"] for s in snapshots],
                "unlinked_cameras": sorted(all_cams - set(cams_T)),
                "unlinked_tags": sorted(
                    all_ids - {tk[0] for tk in tags_T})}

    def world_solve(self, root_id, marker_mm=None):
        """Single-snapshot world solve (the 'Show World Coordinate
        System' button)."""
        marker = float(marker_mm or self.marker_mm)
        obs, views, k_of = self._capture_observations(marker)
        snap = {"obs": obs, "views": views, "k_of": k_of}
        res = self._solve_graph([snap], root_id, marker)
        if res is None:
            return {"ok": False, "views_by_snap": [views],
                    "error": f"Tag {int(root_id)} is not visible to any camera"}
        return res

    # ---------------------------------------- world calibration session
    def wcal_start(self):
        self.wcal = []

    def wcal_snap(self, marker_mm=None):
        marker = float(marker_mm or self.marker_mm)
        obs, views, k_of = self._capture_observations(marker)
        if not hasattr(self, "wcal"):
            self.wcal = []
        self.wcal.append({"obs": obs, "views": views, "k_of": k_of})
        return {"ok": True, "index": len(self.wcal),
                "summary": {str(n): len(t) for n, t in obs.items()}}

    def wcal_solve(self, root_id, marker_mm=None):
        snaps = getattr(self, "wcal", [])
        if not snaps:
            return {"ok": False, "error": "No snapshots captured"}
        marker = float(marker_mm or self.marker_mm)
        res = self._solve_graph(snaps, root_id, marker)
        if res is None:
            return {"ok": False,
                    "views_by_snap": [s["views"] for s in snaps],
                    "error": f"Tag {int(root_id)} is not visible in any snapshot"}
        return res
