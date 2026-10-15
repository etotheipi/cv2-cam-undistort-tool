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
    from . import detectors as det_mod, gestures as gest_mod
    from .tracker import Tracker
except ImportError:          # running as a plain script
    import detectors as det_mod
    import gestures as gest_mod
    from tracker import Tracker


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


PALM = (0, 5, 9, 13, 17)    # wrist + finger bases: the palm's rigid core
HAND_GAP_MM = 60.0          # palm rays of ONE hand meet at least this closely
HAND_TRACK_MM = 150.0       # a hand moves less than this between ticks
HAND_KEEP_S = 0.6           # a hand track survives this long unseen
HAND_SAME_MM = 80.0         # two "hands" with palms this close are one hand
# A tag that drops to one camera keeps the depth of its last multi-camera
# fix for this long (see _fuse): one view measures direction precisely but
# distance poorly.
MV_HOLD_S = 0.6
# (both tuned in simulation: one hand stays whole with ~35 mm of per-view
# palm inconsistency, while two hands 160 mm apart stay separate)
# Handedness vote: a tracked hand never really changes sides, so the vote
# has a long memory and takes sustained contrary evidence to switch.
HAND_SIDE_DECAY = 0.9
HAND_SIDE_SWITCH = 3.0


class OneEuro:
    """One-Euro filter (Casiez et al., CHI 2012) over an (N, 3) array of
    points: heavy smoothing while still, little lag once moving, because
    the cutoff rises with speed. mm and seconds."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt, fc):
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * fc * dt))

    def __call__(self, x, t):
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x
        dt = max(t - self.t, 1e-3)
        ad = self._alpha(dt, self.d_cutoff)
        self.dx = ad * (x - self.x) / dt + (1.0 - ad) * self.dx
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True)
        a = self._alpha(dt, self.min_cutoff + self.beta * speed)
        self.x = a * x + (1.0 - a) * self.x
        self.t = t
        return self.x


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
        self._voter = gest_mod.GestureVoter()
        self._presence = deque(maxlen=3)   # recent hand ids, for ghost filtering
        self._hand_tracks = {}      # track id -> {"X", "t", "L", "side"}
        self._next_hand = 1
        self._filters = {}          # item id -> OneEuro
        self._last_obs = {}         # node -> (frame ts, detections, geom)
        self._mv = {}               # item id -> (time, centre) of last multi-view fix
        # recent hand-grouping evidence (/api/host/live/hand_debug): the
        # real ray gaps are what the thresholds above should be tuned on
        self.hand_debug = deque(maxlen=600)
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
        stylus_cfg = cfg.pop("stylus", None)
        self._stylus = None
        if stylus_cfg and (stylus_cfg.get("calibration") or {}).get("tags"):
            try:
                from . import stylus as _st
            except ImportError:
                import stylus as _st
            self._stylus = _st.StylusModel(stylus_cfg)
            mt = stylus_cfg.get("max_tags")
            # "ml" (default): every corner in every camera; a number: the
            # older fit to the best N triangulated tags
            self._stylus_max = "ml" if mt in (None, "ml") else int(mt)
            self._stylus_prev = None
        self._dets, self._skipped = det_mod.build(detector_keys, **cfg)
        self._span_limit = {d.key: d.max_span_mm for d in self._dets
                            if getattr(d, "max_span_mm", None)}
        self.cams = {}
        self._hand_tracks, self._filters, self._last_obs = {}, {}, {}
        self._mv = {}
        self._presence.clear()
        for node, c in (cams or {}).items():
            node = int(node)
            T = c.get("T_world_cam")
            entry = {"K": c.get("K"), "dist": c.get("dist"),
                     "cal_size": c.get("cal_size"), "T_wc": None,
                     # display rotation: hands are detected upright
                     "rot": int(c.get("rot") or 0) % 360}
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

    def _tag_size(self, oid):
        """A tag's size, learned from earlier multi-camera sightings (None
        if not learned yet; non-tags have no size)."""
        ts = getattr(self, "tag_sizes", None)
        if ts is None or not str(oid).startswith("tag:"):
            return None
        return ts.get(int(str(oid)[4:]))

    def _single_view_center(self, obs, geom, node):
        """Where one camera alone puts this object, in world mm, or None if
        it has no metric single-view model (landmark detectors)."""
        size = self._tag_size(obs.get("id"))
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

    def _steady(self, items):
        """Suppress hand detections that have not held for 2 of the last 3
        frames, and hands only one camera can see.

        MediaPipe mislabels handedness on a rotated hand, so for a frame or
        two a single camera reports a "left hand" that no other camera
        corroborates — a ghost that appears, flickers and vanishes. Both
        filters target that: a hand no second camera sees cannot be placed
        in 3D anyway, and a real hand persists across frames while a
        misclassification does not. Tags are left alone; they carry an id
        that cannot be confused this way.
        """
        keep = []
        for it in items:
            if it["kind"] != "hands":
                keep.append(it)
                continue
            if it.get("n_views", 0) < 2:
                continue                      # one view cannot place a hand
            keep.append(it)
        ids = {i["id"] for i in keep if i["kind"] == "hands"}
        self._presence.append(ids)
        steady = {i for i in ids
                  if sum(1 for f in self._presence if i in f) >= 2}
        return [i for i in keep
                if i["kind"] != "hands" or i["id"] in steady]

    def _fuse(self, obs_by_cam, geoms, frame_ts=None, now=None):
        """obs_by_cam: {node: [obs, ...]} -> list of world-space items.

        Every view of an object is used. Frame capture times are still
        measured and reported (free-running USB cameras drift tens of ms
        apart, which does become position error on a fast-moving object),
        but nothing is discarded over it: dropping views costs more
        accuracy and continuity than the skew costs at normal handling
        speeds, and a hand needs every view it can get.
        """
        frame_ts = frame_ts or {}
        now = time.time() if now is None else now
        grouped, hands = {}, []
        for node, obs in obs_by_cam.items():
            for o in obs:
                if o["kind"] == "hands":
                    if node in geoms:
                        hands.append((node, o))
                    continue
                grouped.setdefault((o["kind"], o["id"]), {})[node] = o
        palm_of = {}
        for k, (by_cam, X) in enumerate(self._hand_groups(hands, geoms)):
            grouped[("hands", f"hand:group{k}")] = by_cam
            palm_of[f"hand:group{k}"] = X
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
            if kind == "hands":
                item["_palm"] = palm_of.get(oid)
                item["_votes"] = [(by_cam[n].get("side"), by_cam[n].get("score"))
                                  for n in nodes]
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
                if kind == "aruco":          # a square's diagonal, with slack
                    limit = 2.2 * (self._tag_size(oid) or 60.0)
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
                self._mv[oid] = (now, np.mean(np.array(good), axis=0))
                # the tag's real size, measured: its sides and diagonals
                ts = getattr(self, "tag_sizes", None)
                if (kind == "aruco" and ts is not None and len(good) == 4 and
                        (not errs or np.sqrt(np.mean(np.square(errs))) < 2.0)):
                    q = np.array(pts3, float)
                    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
                    diag = [np.linalg.norm(q[0] - q[2]), np.linalg.norm(q[1] - q[3])]
                    ts.observe(int(str(oid)[4:]),
                               (np.mean(sides) + np.mean(diag) / np.sqrt(2)) / 2)
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
                # a tag's size is learned from 2+ camera sightings; without
                # it one camera cannot judge distance -- unless the tag was
                # just seen by two, whose range is borrowed below anyway
                size = self._tag_size(oid) if kind == "aruco" else o.get("size_mm")
                recent_mv = oid in self._mv and now - self._mv[oid][0] <= MV_HOLD_S
                if size is None and kind == "aruco" and recent_mv:
                    size = 30.0                  # orientation only; range is borrowed
                if not size or npts != 4:
                    item.update({"points_world": None, "center": None,
                                 "single_view": True, "localized": False,
                                 "reason": ("seen by one camera, size not learned "
                                            "yet — show it to two cameras once"
                                            if kind == "aruco" else
                                            "seen by one camera only — this "
                                            "detector needs two views for 3D")})
                    items.append(item)
                    continue
                g = geoms[n]
                half = float(size) / 2.0
                objp = np.array([[-half, half, 0], [half, half, 0],
                                 [half, -half, 0], [-half, -half, 0]],
                                np.float32)
                pts2 = np.array(o["points"], np.float32)
                sol = Tracker._pnp_square(objp, pts2, g["K"], g["dist"])
                if sol is None:
                    continue
                rvec, tvec, err = sol
                R, _ = cv2.Rodrigues(rvec)
                Tct = np.eye(4)
                Tct[:3, :3] = R
                Tct[:3, 3] = tvec.ravel()
                Tct, err = self._steady_tilt(oid, Tct, err, objp, pts2, g, n, now)
                Twt = self.cams[n]["T_wc"] @ Tct
                wc = (Twt[:3, :3] @ objp.T.astype(np.float64)).T + Twt[:3, 3]
                centre = Twt[:3, 3].copy()
                # Dropped to one camera (a small tag at the edge of a second
                # camera's reach flickers 1-2-1-2). One view gives direction
                # well but depth badly -- depth comes from the apparent size
                # of a small square -- so for a short while keep this
                # camera's direction and the last multi-camera fix's range:
                # slide the tag along the line of sight to it. No added
                # latency, and no jump between the two estimates.
                mv = self._mv.get(oid)
                anchored = False
                if mv is not None and now - mv[0] <= MV_HOLD_S:
                    C = self.cams[n]["T_wc"][:3, 3]
                    d = centre - C
                    d /= np.linalg.norm(d)
                    r = float(d @ (mv[1] - C))
                    if r > 0:
                        shift = C + r * d - centre
                        wc, centre = wc + shift, centre + shift
                        anchored = True
                item.update({
                    "points_world": [[round(float(v), 1) for v in p]
                                     for p in wc],
                    "center": [round(float(v), 1) for v in centre],
                    "rms_px": round(err, 2), "single_view": True,
                    "anchored": anchored, "localized": True,
                })
            item.setdefault("localized", True)
            items.append(item)
        for k in [k for k, (t, _c) in self._mv.items() if now - t > 5.0]:
            del self._mv[k]
        self._track_hands(items, now)
        self._smooth(items, now)
        if getattr(self, "_stylus", None) is not None:
            if self._stylus_max == "ml":
                prev = self._stylus_prev
                loc = self._stylus.locate_ml(
                    items, obs_by_cam, geoms,
                    prev_T=prev[1] if prev and now - prev[0] < 0.5 else None)
            else:
                loc = self._stylus.locate(items, self._stylus_max)
            if loc is not None:
                self._stylus_prev = (now, np.asarray(loc["T"], float))
                items.append(loc)
            sm = getattr(self, "surfmap", None)
            if sm is not None:
                sm.feed(now, loc)            # surface mapping decides what to keep
        for item in items:
            if item["kind"] == "hands" and item.get("points_world"):
                g = gest_mod.classify(item["points_world"])
                if g:
                    item["hand"] = {
                        "scale_mm": g["scale_mm"], "extended": g["extended"],
                        "raw": g["gestures"],
                        # majority vote over a short window: landmark noise
                        # flips borderline predicates frame to frame and a
                        # steady lamp is worth the lag
                        "gestures": self._voter.update(item["id"],
                                                       g["gestures"]),
                    }
        return self._steady(items)

    # ---------------------------------------------------- hands & smoothing
    def _palm_ray(self, o, g):
        """World ray (origin, unit direction) through a detection's palm."""
        pts = np.asarray(o.get("points") or [], np.float64)
        if len(pts) <= max(PALM):
            return None
        px = pts[list(PALM)].mean(axis=0)
        n = cv2.undistortPoints(px.reshape(1, 1, 2), g["K"],
                                g["dist"]).reshape(2)
        R_wc = g["R_cw"].T
        d = R_wc @ np.array([n[0], n[1], 1.0])
        return -R_wc @ g["t_cw"], d / np.linalg.norm(d)

    @staticmethod
    def _meet(rays):
        """Least-squares point nearest a set of rays, its largest distance
        to any of them (mm), and whether it is in front of every camera."""
        A, b = np.zeros((3, 3)), np.zeros(3)
        for C, d in rays:
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ C
        X = np.linalg.lstsq(A, b, rcond=None)[0]
        gap = max(float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C)))
                  for C, d in rays)
        return X, gap, all(float((X - C) @ d) > 0 for C, d in rays)

    def _hand_groups(self, hands, geoms):
        """Which per-camera hand detections are the same physical hand, as
        [({node: obs}, palm point)].

        Handedness cannot decide that: MediaPipe's label is a per-view
        guess that disagrees between cameras and flips on rotated hands, and
        keying on it split one hand into a 'left' and a 'right' copy that
        both triangulated perfectly well. Geometry decides instead: the palm
        rays of one hand meet in a point, two hands' do not. Greedy from the
        tightest pair, at most one detection per camera per hand."""
        hands = self._dedupe_per_camera(hands)
        rays = [self._palm_ray(o, geoms[n]) for n, o in hands]
        pairs = []
        for i in range(len(hands)):
            for j in range(i + 1, len(hands)):
                if (hands[i][0] == hands[j][0] or rays[i] is None
                        or rays[j] is None):
                    continue
                _X, gap, front = self._meet([rays[i], rays[j]])
                if front and gap < HAND_GAP_MM:
                    pairs.append((gap, i, j))
        pairs.sort()
        all_gaps = []
        for i in range(len(hands)):
            for j in range(i + 1, len(hands)):
                if hands[i][0] != hands[j][0] and rays[i] is not None and rays[j] is not None:
                    _X, gap, front = self._meet([rays[i], rays[j]])
                    all_gaps.append([hands[i][0], hands[j][0], round(gap, 1), bool(front)])
        group_of, groups = {}, []
        for _gap, i, j in pairs:
            gi, gj = group_of.get(i), group_of.get(j)
            if gi is None and gj is None:
                group_of[i] = group_of[j] = len(groups)
                groups.append({i, j})
                continue
            if gi is not None and gj is not None:
                # two groups formed from disjoint camera pairs (A+B, C+D)
                # that are really one hand: merge if their rays still meet.
                # Without this, six cameras routinely split one hand in two
                # and both copies triangulated to the same place.
                if gi == gj or not groups[gi] or not groups[gj]:
                    continue
                cams_i = {hands[m][0] for m in groups[gi]}
                if cams_i & {hands[m][0] for m in groups[gj]}:
                    continue
                _X, gap, front = self._meet([rays[m] for m in groups[gi] | groups[gj]])
                if front and gap < HAND_GAP_MM:
                    for m in groups[gj]:
                        group_of[m] = gi
                    groups[gi] |= groups[gj]
                    groups[gj] = set()
                continue
            g, new = (gi, j) if gi is not None else (gj, i)
            if any(hands[m][0] == hands[new][0] for m in groups[g]):
                continue
            _X, gap, front = self._meet([rays[m] for m in groups[g] | {new}])
            if front and gap < HAND_GAP_MM:
                groups[g].add(new)
                group_of[new] = g
        out = sorted(([len(g), {hands[m][0]: hands[m][1] for m in g},
                       self._meet([rays[m] for m in g])[0]] for g in groups if g),
                     key=lambda e: -e[0])
        # a second "hand" whose palm sits on top of a better-supported one
        # is the same hand (no two hands occupy one place)
        keep = []
        for n, by_cam, X in out:
            if all(np.linalg.norm(X - Y) > HAND_SAME_MM for _n, _b, Y in keep):
                keep.append((n, by_cam, X))
        if hands:
            self.hand_debug.append({
                "t": round(time.time(), 3),
                "dets": [[n, o.get("side"), o.get("score")] for n, o in hands],
                "pair_gaps_mm": all_gaps,
                "groups": [sorted(b) for _n, b, _X in out],
                "palms": [[round(float(v), 1) for v in X] for _n, _b, X in out],
                "kept": len(keep)})
        return [(by_cam, X) for _n, by_cam, X in keep]

    @staticmethod
    def _dedupe_per_camera(hands):
        """MediaPipe sometimes reports one hand twice in a frame. Within each
        camera, a detection whose palm lies within half a hand-width of a
        stronger one is that same hand."""
        out = []
        for node, o in sorted(hands, key=lambda h: -(h[1].get("score") or 0)):
            pts = np.asarray(o.get("points") or [], np.float64)
            if len(pts) <= max(PALM):
                continue
            c = pts[list(PALM)].mean(axis=0)
            size = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            if any(n2 == node and np.linalg.norm(c - c2) < 0.5 * max(size, s2)
                   for n2, _o2, c2, s2 in out):
                continue
            out.append((node, o, c, size))
        return [(n, o) for n, o, _c, _s in out]

    def _track_hands(self, items, now):
        """Stable identity and handedness for fused hands. A hand keeps its
        track id while it moves less than HAND_TRACK_MM per tick; its side
        is a decaying, score-weighted vote over every camera's label with
        hysteresis, so one camera -- or one frame -- changing its mind no
        longer flips the hand."""
        tracks = self._hand_tracks
        for tid in [t for t, tr in tracks.items() if now - tr["t"] > HAND_KEEP_S]:
            del tracks[tid]
        hs = [it for it in items if it["kind"] == "hands"]
        where = [it.get("center") if it.get("center") is not None
                 else it.get("_palm") for it in hs]
        cand = sorted((float(np.linalg.norm(np.subtract(w, tr["X"]))), k, tid)
                      for k, w in enumerate(where) if w is not None
                      for tid, tr in tracks.items())
        match, used = {}, set()
        for dist, k, tid in cand:
            if dist > HAND_TRACK_MM:
                break
            if k not in match and tid not in used:
                match[k] = tid
                used.add(tid)
        for k, it in enumerate(hs):
            votes = it.pop("_votes", [])
            it.pop("_palm", None)
            if where[k] is None:
                continue
            tid = match.get(k)
            if tid is None:
                tid = self._next_hand
                self._next_hand += 1
                tracks[tid] = {"L": 0.0, "side": None}
            tr = tracks[tid]
            tr["X"], tr["t"] = np.asarray(where[k], float), now
            tr["L"] = HAND_SIDE_DECAY * tr["L"] + sum(
                (s or 0.5) * (1 if side == "right" else -1 if side == "left" else 0)
                for side, s in votes)
            if tr["side"] is None:
                # a new hand commits only on clear evidence (a split vote
                # waits a frame or two) -- a wrong first guess is what
                # would otherwise need correcting later
                if abs(tr["L"]) >= 1.5:
                    tr["side"] = "right" if tr["L"] > 0 else "left"
            elif tr["side"] == "right" and tr["L"] < -HAND_SIDE_SWITCH:
                tr["side"] = "left"
            elif tr["side"] == "left" and tr["L"] > HAND_SIDE_SWITCH:
                tr["side"] = "right"
            it.update({"id": f"hand:{tid}",
                       "label": f"{tr['side']} hand" if tr["side"] else "hand",
                       "side": tr["side"]})

    def _steady_tilt(self, oid, Tct, err, objp, pts, g, node, now):
        """A tag seen by one camera fits two tilts about equally (IPPE's
        ambiguity), and the better-fitting one can alternate frame to frame
        -- the tag visibly flips. Keep whichever candidate is closest to
        where the tag just was, among those that still fit the corners."""
        f = self._filters.get(oid)
        if f is None or f.x is None or f.x.shape != (4, 3) or now - f.t > 0.5:
            return Tct, err
        T_wc = self.cams[node]["T_wc"]
        best = None
        for T in [Tct] + Tracker._ippe_candidates(objp.astype(np.float64), pts,
                                                  g["K"], g["dist"]):
            e = float(np.sqrt(Tracker._reproj_sq(T, objp.astype(np.float64),
                                                 pts, g["K"], g["dist"]) / 4))
            if not np.isfinite(e) or e > 2.0 * err + 1.0:
                continue
            Tw = T_wc @ T
            wc = (Tw[:3, :3] @ objp.T.astype(np.float64)).T + Tw[:3, 3]
            d = float(np.linalg.norm(wc - f.x, axis=1).sum())
            if best is None or d < best[0]:
                best = (d, T, e)
        return (best[1], round(best[2], 3)) if best else (Tct, err)

    def _smooth(self, items, now):
        """One-Euro filtering of every localized item, keyed by its (now
        stable) id: still objects stop shimmering, moving ones barely lag."""
        for it in items:
            pw = it.get("points_world")
            if not pw or any(p is None for p in pw):
                continue
            f = self._filters.get(it["id"])
            if f is None:
                f = self._filters[it["id"]] = OneEuro()
            elif f.t is not None and now - f.t > 0.5:
                f.x = None                  # reappeared: start afresh
            sm = f(np.asarray(pw, np.float64), now)
            it["points_world"] = [[round(float(v), 1) for v in p] for p in sm]
            it["center"] = [round(float(v), 1) for v in sm.mean(axis=0)]
        for k in [k for k, f in self._filters.items()
                  if f.t is not None and now - f.t > 2.0]:
            del self._filters[k]

    # ---------------------------------------------------------------- loop
    def _detect_one(self, job):
        """One camera's full detector sweep. Runs on a pool thread; touches
        only its own camera's state, and returns rather than mutating
        shared dicts so the caller does the merging single-threaded."""
        node, frame, w, h, ts = job
        got, ms, err = [], {}, None
        meta = {"node": node, "size": (w, h), "ts": ts,
                "rot": (self.cams.get(node) or {}).get("rot", 0)}
        for d in self._dets:
            t1 = time.time()
            try:
                got.extend(d.detect(frame, meta))
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
                jobs.append((node, frame, w, h, ts))

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
            # A camera whose frame has not advanced this tick keeps its last
            # detections while they are fresh. Capture and tracking run at
            # similar rates, so otherwise cameras drop in and out tick to
            # tick, the set of views an object is fused from keeps changing,
            # and every change is a small jump.
            fresh_s = max(0.15, 1.5 / max(self.track_fps, 0.1))
            for node in list(self._last_obs):
                ts_n, got_n, g_n = self._last_obs[node]
                if node in obs_by_cam:
                    continue
                if time.time() - ts_n <= fresh_s and node in self.cams:
                    obs_by_cam[node], geoms[node] = got_n, g_n
                    frame_ts[node] = ts_n
                else:
                    del self._last_obs[node]
            for node, got, _ms, _err in results:
                self._last_obs[node] = (frame_ts[node], got, geoms[node])

            items = self._fuse(obs_by_cam, geoms, frame_ts, now=t0)
            self._voter.forget({i["id"] for i in items})
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
