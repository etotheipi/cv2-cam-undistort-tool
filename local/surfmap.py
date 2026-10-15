"""Surface mapping with a calibrated stylus.

The live tracker feeds every stylus fix (feed()); this decides when a fix
becomes a surface sample, according to the capture mode:

  drag   while the operator holds the trigger (Space / a foot pedal), keep a
         point every `spacing` mm of travel -- but only from moments when
         the tip is moving slower than `max_speed` (a fast tip is a blurred,
         time-skewed one), each the mean of the last `window` seconds
  dwell  hands-free: a point whenever the tip has rested still (spread
         under `still` mm) for `dwell` seconds at a new spot
  tap    the trigger takes one point: the mean of the next 0.5 s

Each sample keeps what the normals need later: the ball centre, the stylus
axis (which side of the surface the stylus was on), and quality figures.

compute() then estimates a normal per sample from its neighbours (PCA of
the local patch, oriented toward the stylus), moves each centre one ball
radius back along it to the surface contact point, and meshes the contact
points by Delaunay triangulation over their best-fit plane (a height-field
surface; a closed object needs Poisson reconstruction elsewhere -- export
the PLY). Exports: PLY (points + normals + faces), OBJ, STL, CSV.
"""

import io
import json
import math
import pathlib
import threading
import time
from collections import deque

import numpy as np

STORE = pathlib.Path(__file__).resolve().parent / "maps" / "session.json"


class SurfaceMap:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = {"mode": "drag", "spacing_mm": 1.0, "max_speed_mm_s": 25.0,
                    "window_s": 0.2, "still_mm": 0.35, "dwell_s": 0.4}
        self.points = []
        self.buf = deque()            # (t, tip, axis, fit_px, cams, ntags)
        self.armed = False
        self.tap_t0 = None
        self.dwell_done = False       # one sample per rest
        self.origin = None
        self.radius = 0.0
        self.result = None
        self.state = {}
        self._load()

    # ---------------------------------------------------------- control
    def configure(self, **kw):
        with self.lock:
            for k, v in kw.items():
                if k in self.cfg and v is not None:
                    self.cfg[k] = v if k == "mode" else float(v)

    def arm(self, on):
        with self.lock:
            self.armed = bool(on)

    def tap(self):
        with self.lock:
            self.tap_t0 = time.time()

    def undo(self):
        with self.lock:
            if self.points:
                self.points.pop()
                self.result = None
                self._save()

    def clear(self):
        with self.lock:
            self.points, self.result = [], None
            self._save()

    # ------------------------------------------------------------- feed
    def feed(self, now, loc):
        """One live-tracker tick: loc is the stylus item or None."""
        with self.lock:
            if loc is None:
                self.buf.clear()                       # tip lost: no averaging across gaps
                self.state = {"visible": False, "armed": self.armed}
                return
            self.origin = loc.get("origin")
            self.radius = float(loc.get("radius_mm") or 0)
            self.buf.append((now, np.asarray(loc["tip"], float),
                             np.asarray(loc["axis"], float),
                             float(loc.get("fit_px") or loc.get("fit_mm") or 0),
                             int(loc.get("cams_used") or loc.get("n_views") or 0),
                             len(loc.get("tags_used") or [])))
            while self.buf and now - self.buf[0][0] > 1.5:
                self.buf.popleft()
            c = self.cfg
            recent = [b for b in self.buf if now - b[0] <= c["window_s"]]
            speed = None
            if len(self.buf) >= 2:
                old = [b for b in self.buf if now - b[0] >= c["window_s"]]
                ref = old[-1] if old else self.buf[0]
                dt = now - ref[0]
                if dt > 1e-3:
                    speed = float(np.linalg.norm(self.buf[-1][1] - ref[1]) / dt)
            last = self.points[-1]["center"] if self.points else None

            def far(win):
                # spacing between SAMPLES: compare the window mean that would
                # be kept (it lags the live tip), not the live tip itself
                if last is None or not win:
                    return True
                m = np.mean([b[1] for b in win], axis=0)
                return np.linalg.norm(m - np.asarray(last)) >= c["spacing_mm"]
            mode = c["mode"]
            took = None
            if mode == "drag" and self.armed and len(recent) >= 2 and far(recent):
                if speed is not None and speed <= c["max_speed_mm_s"]:
                    took = self._capture(recent, speed, mode)
            elif mode == "dwell":
                rest = [b for b in self.buf if now - b[0] <= c["dwell_s"]]
                still = False
                if len(rest) >= 3 and now - rest[0][0] >= 0.8 * c["dwell_s"]:
                    P = np.array([b[1] for b in rest])
                    still = float(np.linalg.norm(P - P.mean(0), axis=1).max()) <= c["still_mm"]
                if not still:
                    self.dwell_done = False
                elif not self.dwell_done and far(rest):
                    took = self._capture(rest, speed, mode)
                    self.dwell_done = True
            elif mode == "tap" and self.tap_t0 is not None and now - self.tap_t0 >= 0.5:
                win = [b for b in self.buf if b[0] >= self.tap_t0]
                self.tap_t0 = None
                if len(win) >= 2:
                    took = self._capture(win, speed, mode)
            self.state = {"visible": True, "armed": self.armed,
                          "speed_mm_s": round(speed, 1) if speed is not None else None,
                          "tip": [round(float(v), 2) for v in self.buf[-1][1]],
                          "captured": took is not None}

    def _capture(self, win, speed, mode):
        P = np.array([b[1] for b in win])
        A = np.array([b[2] for b in win]).mean(0)
        A /= np.linalg.norm(A)
        ctr = P.mean(0)
        p = {"i": len(self.points), "t": round(win[-1][0], 3), "mode": mode,
             "center": [round(float(v), 3) for v in ctr],
             "axis": [round(float(v), 5) for v in A],
             "sd_mm": round(float(np.sqrt(np.mean(np.sum((P - ctr) ** 2, 1)))), 3),
             "n": len(win),
             "speed_mm_s": round(float(speed), 2) if speed is not None else None,
             "fit": round(float(np.mean([b[3] for b in win])), 3),
             "cams": int(min(b[4] for b in win)), "tags": int(min(b[5] for b in win))}
        self.points.append(p)
        self.result = None
        self._save()
        return p

    def snapshot(self, since=0):
        with self.lock:
            return {"count": len(self.points), "points": self.points[since:],
                    "state": dict(self.state), "cfg": dict(self.cfg),
                    "origin": self.origin, "radius_mm": self.radius,
                    "has_result": self.result is not None}

    # ------------------------------------------------------- processing
    def compute(self, k=10, max_edge_factor=3.0):
        """Normals (local PCA, oriented toward the stylus), ball-radius
        compensated contact points, and a Delaunay mesh over the best-fit
        plane with over-long triangles removed."""
        from scipy.spatial import cKDTree, Delaunay
        with self.lock:
            pts = list(self.points)
            r = self.radius
        if len(pts) < 4:
            return {"ok": False, "error": "need at least 4 points"}
        C = np.array([p["center"] for p in pts])
        Ax = np.array([p["axis"] for p in pts])
        kk = int(min(max(k, 4), len(C)))
        tree = cKDTree(C)
        _d, nb = tree.query(C, k=kk)
        N, flat = np.zeros_like(C), np.zeros(len(C))
        for i, idx in enumerate(nb):
            Q = C[idx] - C[idx].mean(0)
            w, v = np.linalg.eigh(Q.T @ Q)
            n = v[:, 0]
            if n @ Ax[i] < 0:
                n = -n                                  # out of the surface, toward the stylus
            N[i] = n
            flat[i] = w[0] / max(w.sum(), 1e-12)       # 0 = perfectly flat patch
        S = C - r * N                                   # surface contact points
        # mesh: Delaunay over the global best-fit plane
        m = S.mean(0)
        _w, V = np.linalg.eigh((S - m).T @ (S - m))
        uv = (S - m) @ V[:, 1:]
        faces = []
        if len(S) >= 3:
            tri = Delaunay(uv)
            # scale from the 4th-nearest neighbour: along a drag line points
            # are dense, ACROSS lines they are as far apart as the lines are
            k4 = min(5, len(C))
            nn = tree.query(C, k=k4)[0][:, -1]
            max_edge = max_edge_factor * float(np.median(nn))
            for a, b, c in tri.simplices:
                e = max(np.linalg.norm(S[a] - S[b]), np.linalg.norm(S[b] - S[c]),
                        np.linalg.norm(S[c] - S[a]))
                if e > max_edge:
                    continue
                fn = np.cross(S[b] - S[a], S[c] - S[a])
                if np.linalg.norm(fn) < 1e-9:
                    continue
                fn /= np.linalg.norm(fn)
                vn = N[a] + N[b] + N[c]
                if fn @ vn < 0:                         # wind consistently with the normals
                    b, c = c, b
                    fn = -fn
                if fn @ (vn / np.linalg.norm(vn)) < 0.5:  # a fold, not the surface
                    continue
                faces.append([int(a), int(b), int(c)])
        res = {"ok": True, "surface": S.round(3).tolist(), "normals": N.round(5).tolist(),
               "flatness": flat.round(5).tolist(), "faces": faces,
               "stats": {"points": len(S), "faces": len(faces), "ball_radius_mm": r,
                         "median_spacing_mm": round(float(np.median(tree.query(C, k=2)[0][:, 1])), 3),
                         "median_sample_sd_mm": round(float(np.median([p["sd_mm"] for p in pts])), 3)}}
        with self.lock:
            self.result = res
        return res

    # ----------------------------------------------------------- export
    def export(self, fmt, relative=True):
        """(filename, mime, bytes). Coordinates in mm, world frame -- or,
        relative=True, relative to the stylus calibration divot."""
        with self.lock:
            pts, res, origin = list(self.points), self.result, self.origin
        if res is None and fmt != "csv":
            res = self.compute()
            if not res.get("ok"):
                raise ValueError(res.get("error"))
        off = np.asarray(origin, float) if (relative and origin) else np.zeros(3)
        frame = "relative to the stylus calibration divot" if off.any() else "world frame"
        if fmt == "csv":
            out = io.StringIO()
            out.write("# stylus surface samples, mm, " + frame + "\n")
            out.write("i,t,mode,cx,cy,cz,ax,ay,az,sd_mm,n,speed_mm_s,fit,cams,tags\n")
            for p in pts:
                c = np.asarray(p["center"]) - off
                out.write(",".join(map(str, [p["i"], p["t"], p["mode"], *np.round(c, 3),
                                             *p["axis"], p["sd_mm"], p["n"], p["speed_mm_s"],
                                             p["fit"], p["cams"], p["tags"]])) + "\n")
            return "surface_samples.csv", "text/csv", out.getvalue().encode()
        S = np.asarray(res["surface"]) - off
        N = np.asarray(res["normals"])
        F = res["faces"]
        if fmt == "ply":
            out = io.StringIO()
            out.write("ply\nformat ascii 1.0\ncomment stylus surface map, mm, " + frame +
                      "\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                      "property float nx\nproperty float ny\nproperty float nz\n" % len(S))
            out.write("element face %d\nproperty list uchar int vertex_indices\nend_header\n" % len(F))
            for s, n in zip(S, N):
                out.write("%.4f %.4f %.4f %.5f %.5f %.5f\n" % (*s, *n))
            for f in F:
                out.write("3 %d %d %d\n" % tuple(f))
            return "surface.ply", "application/octet-stream", out.getvalue().encode()
        if fmt == "obj":
            out = io.StringIO()
            out.write("# stylus surface map, mm, " + frame + "\n")
            for s in S:
                out.write("v %.4f %.4f %.4f\n" % tuple(s))
            for n in N:
                out.write("vn %.5f %.5f %.5f\n" % tuple(n))
            for a, b, c in F:
                out.write("f %d//%d %d//%d %d//%d\n" % (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1))
            return "surface.obj", "text/plain", out.getvalue().encode()
        if fmt == "stl":
            out = io.StringIO()
            out.write("solid stylus_surface\n")
            for a, b, c in F:
                fn = np.cross(S[b] - S[a], S[c] - S[a])
                fn = fn / (np.linalg.norm(fn) or 1)
                out.write(" facet normal %.5f %.5f %.5f\n  outer loop\n" % tuple(fn))
                for v in (S[a], S[b], S[c]):
                    out.write("   vertex %.4f %.4f %.4f\n" % tuple(v))
                out.write("  endloop\n endfacet\n")
            out.write("endsolid stylus_surface\n")
            return "surface.stl", "model/stl", out.getvalue().encode()
        raise ValueError("unknown format " + fmt)

    # ---------------------------------------------------------- persist
    def _save(self):
        try:
            STORE.parent.mkdir(parents=True, exist_ok=True)
            STORE.write_text(json.dumps({"points": self.points, "cfg": self.cfg}))
        except OSError:
            pass

    def _load(self):
        try:
            d = json.loads(STORE.read_text())
            self.points = d.get("points") or []
            self.cfg.update(d.get("cfg") or {})
        except (OSError, ValueError):
            pass
