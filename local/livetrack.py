"""Live multi-camera tracking: detectors in, world-space items out.

Each tick sweeps the selected cameras, runs the selected detectors on the
newest frame from each, and fuses the per-camera observations into world
coordinates using the cameras' saved extrinsics.

Fusion rule, per observation id:

  >= 2 cameras   triangulate every point index independently (DLT over all
                 views that saw it). Works for anything with a stable point
                 order — tag corners, hand landmarks — and needs no metric
                 model. Reports the reprojection RMS so the UI can show how
                 well the views agreed.

  1 camera       only possible when the detector recovers a metric pose from
                 a single view. Tags do (planar, known size); landmark
                 models do not, so a hand seen by one camera is reported as
                 detected-but-not-localizable rather than guessed at.

Point k in one camera must be point k in another — that is the whole
contract the detector interface exists to enforce.
"""

import os
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

try:
    from . import detectors as det_mod
    from .tracker import Tracker
except ImportError:          # running as a plain script
    import detectors as det_mod
    from tracker import Tracker


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


class GpuMetrics:
    """nvidia-smi, sampled at most every `period` seconds.

    Spawning a process per poll would cost more than the tracker loop, so
    results are cached; absence of nvidia-smi is a normal state, not an
    error, and is reported once as unavailable.
    """

    def __init__(self, period=2.0):
        self.period = float(period)
        self._last_t = 0.0
        self._cache = None
        self._available = None

    def sample(self):
        now = time.time()
        if self._cache is not None and now - self._last_t < self.period:
            return self._cache
        self._last_t = now
        if self._available is False:
            return self._cache
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,"
                 "memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            self._available = False
            self._cache = {"available": False,
                           "reason": "nvidia-smi not present"}
            return self._cache
        if r.returncode != 0:
            self._available = False
            self._cache = {"available": False,
                           "reason": (r.stderr or "nvidia-smi failed").strip()[:120]}
            return self._cache

        def num(s):
            try:
                return float(s)
            except ValueError:
                return None

        gpus = []
        for line in r.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            used, total = num(f[3]), num(f[4])
            gpus.append({
                "index": f[0], "name": f[1],
                "util_pct": num(f[2]),
                "mem_used_mb": used, "mem_total_mb": total,
                "mem_pct": (round(100.0 * used / total, 1)
                            if used is not None and total else None),
                "temp_c": num(f[5]), "power_w": num(f[6]),
            })
        self._available = bool(gpus)
        self._cache = {"available": bool(gpus), "gpus": gpus,
                       "reason": None if gpus else "no GPUs reported"}
        return self._cache


class LiveTracker:
    def __init__(self, streams):
        self.streams = streams
        self.gpu = GpuMetrics()
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._dets = []
        self._skipped = []
        self._pool = None
        self._span_limit = {}
        self.cams = {}              # node -> {"K","dist","cal_size","T_wc"}
        self.track_fps = 10.0
        self.warmup_s = 1.5
        self.items = []
        self.per_cam = {}
        self.stats = {}
        self.seq = 0

    # ------------------------------------------------------------ control
    def start(self, cams, detector_keys, track_fps=10.0, warmup_s=1.5, **cfg):
        """cams: {node: {"K","dist","cal_size","T_world_cam"}}."""
        self.stop()
        cfg.pop("sync_ms", None)   # no longer used: all views are fused
        self._dets, self._skipped = det_mod.build(detector_keys, **cfg)
        self._span_limit = {d.key: d.max_span_mm for d in self._dets
                            if getattr(d, "max_span_mm", None)}
        self.cams = {}
        for node, c in (cams or {}).items():
            node = int(node)
            T = c.get("T_world_cam")
            entry = {"K": c.get("K"), "dist": c.get("dist"),
                     "cal_size": c.get("cal_size"), "T_wc": None}
            if T is not None:
                try:
                    entry["T_wc"] = np.array(T, np.float64).reshape(4, 4)
                except (ValueError, TypeError):
                    entry["T_wc"] = None
            self.cams[node] = entry
        self.track_fps = float(track_fps or 10.0)
        self.warmup_s = float(warmup_s)
        self.items, self.per_cam, self.stats = [], {}, {}
        self.seq = 0
        if not self.cams or not self._dets:
            return {"ok": False,
                    "error": ("no runnable detector" if not self._dets
                              else "no cameras"),
                    "skipped": self._skipped}
        # Detection is per-camera independent and releases the GIL inside
        # both OpenCV and mediapipe, so cameras run concurrently instead of
        # one after another. Serial cost grows linearly with camera count;
        # this keeps it nearly flat until the cores run out. Leave a couple
        # of cores for capture threads and the rest of the server.
        self._workers = max(1, min(len(self.cams), (os.cpu_count() or 4) - 2))
        self._pool = ThreadPoolExecutor(max_workers=self._workers,
                                        thread_name_prefix="livedet")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"ok": True, "detectors": [d.key for d in self._dets],
                "skipped": self._skipped, "cameras": sorted(self.cams),
                "workers": self._workers}

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.5)
            self._thread = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for d in self._dets:
            try:
                d.close()
            except Exception:
                pass
        self._dets = []

    @property
    def running(self):
        return self._running

    # ------------------------------------------------------------ geometry
    def _K_for(self, node, w, h):
        c = self.cams.get(node) or {}
        if c.get("K"):
            cw, ch = c.get("cal_size") or (w, h)
            K = np.array(c["K"], np.float64)
            if (cw, ch) != (w, h):
                K = np.diag([w / cw, h / ch, 1.0]) @ K
            return K, np.array(c.get("dist") or [0] * 5, np.float64)
        f = 0.8 * w
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                        np.float64), np.zeros(5)

    def _geom(self, node, w, h):
        """(K, dist, P 3x4 world->cam normalized, T_cam_world) or None."""
        c = self.cams.get(node)
        if c is None or c.get("T_wc") is None:
            return None
        K, dist = self._K_for(node, w, h)
        Tcw = _inv(c["T_wc"])
        return {"K": K, "dist": dist,
                "P": np.hstack([Tcw[:3, :3], Tcw[:3, 3:4]]),
                "R_cw": Tcw[:3, :3], "t_cw": Tcw[:3, 3]}

    def _single_view_center(self, obs, geom, node):
        """Where one camera alone puts this object, in world mm, or None if
        it has no metric single-view model (landmark detectors)."""
        size = obs.get("size_mm")
        if not size or len(obs.get("points") or []) != 4:
            return None
        half = float(size) / 2.0
        objp = np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], np.float32)
        sol = Tracker._pnp_square(objp, np.array(obs["points"], np.float32),
                                  geom["K"], geom["dist"])
        if sol is None:
            return None
        R, _ = cv2.Rodrigues(sol[0])
        Tct = np.eye(4)
        Tct[:3, :3] = R
        Tct[:3, 3] = sol[1].ravel()
        return (self.cams[node]["T_wc"] @ Tct)[:3, 3]

    def _span(self, pts):
        """Largest distance between any two reconstructed points, mm."""
        good = np.array([p for p in pts if p is not None], float)
        if len(good) < 2:
            return 0.0
        d = good[:, None, :] - good[None, :, :]
        return float(np.sqrt((d * d).sum(-1)).max())

    def _fuse(self, obs_by_cam, geoms, frame_ts=None):
        """obs_by_cam: {node: [obs, ...]} -> list of world-space items.

        Every view of an object is used. Frame capture times are still
        measured and reported (free-running USB cameras drift tens of ms
        apart, which does become position error on a fast-moving object),
        but nothing is discarded over it: dropping views costs more
        accuracy and continuity than the skew costs at normal handling
        speeds, and a hand needs every view it can get.
        """
        frame_ts = frame_ts or {}
        grouped = {}
        for node, obs in obs_by_cam.items():
            for o in obs:
                grouped.setdefault((o["kind"], o["id"]), {})[node] = o
        items = []
        for (kind, oid), by_cam in grouped.items():
            nodes = [n for n in by_cam if n in geoms]
            if not nodes:
                continue
            ts = [frame_ts[n] for n in nodes if n in frame_ts]
            skew_ms = (round((max(ts) - min(ts)) * 1000.0, 1)
                       if len(ts) >= 2 else 0)
            sample = by_cam[nodes[0]]
            npts = len(sample["points"])
            item = {"kind": kind, "id": oid, "label": sample.get("label", oid),
                    "cameras": sorted(nodes), "n_views": len(nodes),
                    "names": sample.get("names"), "skew_ms": skew_ms}
            if len(nodes) >= 2:
                pts3, errs = [], []
                for k in range(npts):
                    rays, seen = [], []
                    for n in nodes:
                        o = by_cam[n]
                        if k >= len(o["points"]):
                            continue
                        g = geoms[n]
                        und = cv2.undistortPoints(
                            np.array(o["points"][k], np.float64).reshape(1, 1, 2),
                            g["K"], g["dist"]).reshape(2)
                        rays.append((g["P"], und[0], und[1]))
                        seen.append((n, o["points"][k]))
                    X = Tracker._triangulate(rays) if len(rays) >= 2 else None
                    if X is None:
                        pts3.append(None)
                        continue
                    pts3.append([round(float(v), 1) for v in X])
                    for n, px in seen:
                        g = geoms[n]
                        rv, _ = cv2.Rodrigues(g["R_cw"])
                        proj, _ = cv2.projectPoints(
                            X.reshape(1, 3), rv, g["t_cw"], g["K"], g["dist"])
                        errs.append(float(np.linalg.norm(
                            proj.reshape(2) - np.array(px, np.float64))))
                good = [p for p in pts3 if p is not None]
                if not good:
                    continue
                # Structural sanity. Two views cannot tell a correct pairing
                # from a wrong one -- any two rays meet, so a mismatched
                # correspondence (mediapipe flipping handedness for one
                # frame, say) reprojects perfectly while placing the object
                # metres away. Only a prior on the object's real size
                # catches that, and it is what stops the one-frame spikes.
                span = self._span(pts3)
                limit = self._span_limit.get(kind)
                if limit and span > limit:
                    item.update({
                        "points_world": None, "center": None,
                        "single_view": False, "localized": False,
                        "span_mm": round(span, 1),
                        "reason": (f"reconstruction spans {span:.0f} mm, over "
                                   f"the {limit:.0f} mm limit for {kind} — "
                                   "views disagree about which object this "
                                   "is, so the frame is dropped"),
                    })
                    items.append(item)
                    continue
                item.update({
                    "points_world": pts3,
                    "center": [round(float(v), 1)
                               for v in np.mean(np.array(good), axis=0)],
                    "rms_px": (round(float(np.sqrt(np.mean(np.square(errs)))), 2)
                               if errs else None),
                    "single_view": False,
                })
                # Where each camera would put this object on its own. When
                # the fused result looks wrong, this says whether one
                # camera disagrees (its extrinsic drifted) or the views
                # simply cannot pin it down (grazing angle, tiny in frame).
                pv = {}
                for n in nodes:
                    c = self._single_view_center(by_cam[n], geoms[n], n)
                    if c is not None:
                        pv[str(n)] = [round(float(v), 1) for v in c]
                if len(pv) >= 2:
                    P = np.array(list(pv.values()))
                    item["view_spread_mm"] = round(float(np.max(
                        np.linalg.norm(P - P.mean(axis=0), axis=1)) * 2), 1)
                if pv:
                    item["per_view_center"] = pv
            else:
                # one view: only a metric single-view model can be placed
                n = nodes[0]
                o = by_cam[n]
                size = o.get("size_mm")
                if not size or npts != 4:
                    item.update({"points_world": None, "center": None,
                                 "single_view": True, "localized": False,
                                 "reason": ("seen by one camera only — this "
                                            "detector needs two views for 3D")})
                    items.append(item)
                    continue
                g = geoms[n]
                half = float(size) / 2.0
                objp = np.array([[-half, half, 0], [half, half, 0],
                                 [half, -half, 0], [-half, -half, 0]],
                                np.float32)
                sol = Tracker._pnp_square(
                    objp, np.array(o["points"], np.float32), g["K"], g["dist"])
                if sol is None:
                    continue
                rvec, tvec, err = sol
                R, _ = cv2.Rodrigues(rvec)
                Tct = np.eye(4)
                Tct[:3, :3] = R
                Tct[:3, 3] = tvec.ravel()
                Twt = self.cams[n]["T_wc"] @ Tct
                wc = (Twt[:3, :3] @ objp.T.astype(np.float64)).T + Twt[:3, 3]
                item.update({
                    "points_world": [[round(float(v), 1) for v in p]
                                     for p in wc],
                    "center": [round(float(v), 1) for v in Twt[:3, 3]],
                    "rms_px": round(err, 2), "single_view": True,
                    "localized": True,
                })
            item.setdefault("localized", True)
            items.append(item)
        return items

    # ---------------------------------------------------------------- loop
    def _detect_one(self, job):
        """One camera's full detector sweep. Runs on a pool thread; touches
        only its own camera's state, and returns rather than mutating
        shared dicts so the caller does the merging single-threaded."""
        node, frame, w, h = job
        got, ms, err = [], {}, None
        for d in self._dets:
            t1 = time.time()
            try:
                got.extend(d.detect(frame, {"node": node, "size": (w, h)}))
            except Exception as e:
                err = f"{d.key}: {e.__class__.__name__}: {e}"
            ms[d.key] = round((time.time() - t1) * 1000.0, 2)
        return node, got, ms, err

    def _loop(self):
        time.sleep(self.warmup_s)
        seqs = {}
        ticks = deque(maxlen=20)
        busys = deque(maxlen=20)
        walls = deque(maxlen=20)
        skews = deque(maxlen=20)
        det_ms = {d.key: deque(maxlen=30) for d in self._dets}
        while self._running:
            t0 = time.time()
            obs_by_cam, geoms, per_cam, jobs, sizes = {}, {}, {}, [], {}
            frame_ts = {}
            # frame grab stays serial: it is a memcpy from the capture
            # thread, and the sequence bookkeeping is this thread's
            for node in list(self.cams):
                if not self._running:
                    return
                st = self.streams.get(node)
                if st is None or not st.started:
                    per_cam[node] = {"error": "no stream"}
                    continue
                frame, seq, ts = st.get_frame(seqs.get(node, 0), timeout=0.005)
                if frame is None or seq == seqs.get(node):
                    continue
                seqs[node] = seq
                frame_ts[node] = ts
                h, w = frame.shape[:2]
                g = self._geom(node, w, h)
                if g is None:
                    per_cam[node] = {"error": "no world pose"}
                    continue
                geoms[node] = g
                sizes[node] = [w, h]
                jobs.append((node, frame, w, h))

            t_det = time.time()
            results = list(self._pool.map(self._detect_one, jobs)) if jobs \
                else []
            wall = (time.time() - t_det) * 1000.0
            for node, got, ms, err in results:
                obs_by_cam[node] = got
                e = per_cam.setdefault(node, {})
                e.update({"n": len(got), "detect_ms": ms,
                          "size": sizes.get(node)})
                if err:
                    e["error"] = err
                for k, v in ms.items():
                    det_ms[k].append(v)
            if jobs:
                walls.append(wall)

            items = self._fuse(obs_by_cam, geoms, frame_ts)
            if len(frame_ts) >= 2:
                sk = (max(frame_ts.values()) - min(frame_ts.values())) * 1000.0
                skews.append(sk)
            busy = time.time() - t0
            ticks.append(t0)
            busys.append(busy)
            with self._lock:
                self.items = items
                self.per_cam = per_cam
                self.seq += 1
                stats = {"detectors": {
                    k: round(float(np.mean(v)), 2) for k, v in det_ms.items()
                    if v}}
                if len(ticks) >= 2 and ticks[-1] > ticks[0]:
                    span = ticks[-1] - ticks[0]
                    stats["achieved_fps"] = round((len(ticks) - 1) / span, 2)
                    stats["duty_pct"] = round(
                        100.0 * sum(busys) / (span + busy), 1)
                stats["target_fps"] = self.track_fps
                stats["workers"] = self._workers
                if skews:
                    stats["frame_skew_ms"] = round(float(np.mean(skews)), 1)
                    stats["frame_skew_max_ms"] = round(float(max(skews)), 1)
                if walls:
                    # wall time of the parallel detection phase, against
                    # what the same work would cost run back to back —
                    # makes the benefit of the pool visible rather than
                    # something you have to take on faith
                    stats["detect_wall_ms"] = round(float(np.mean(walls)), 2)
                    serial = sum(float(np.mean(v)) for v in det_ms.values()
                                 if v) * max(1, len(jobs))
                    stats["detect_serial_ms"] = round(serial, 2)
                    if stats["detect_wall_ms"] > 0:
                        stats["parallel_speedup"] = round(
                            serial / stats["detect_wall_ms"], 2)
                stats["items"] = len(items)
                stats["rejected"] = sum(
                    1 for i in items if i.get("localized") is False
                    and i.get("span_mm") is not None)
                stats["localized"] = sum(
                    1 for i in items if i.get("localized"))
                self.stats = stats
            time.sleep(max(0.0, 1.0 / max(self.track_fps, 0.1) - busy))

    def snapshot(self):
        with self._lock:
            return {"running": self._running, "seq": self.seq,
                    "items": list(self.items), "per_cam": dict(self.per_cam),
                    "stats": dict(self.stats),
                    "detectors": [d.key for d in self._dets],
                    "skipped": list(self._skipped),
                    "gpu": self.gpu.sample()}
