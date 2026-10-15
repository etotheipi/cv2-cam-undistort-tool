"""Learned tag sizes, per tag ID.

Once the cameras are calibrated, a tag seen by two or more of them is
triangulated corner by corner, so its real size falls out of the geometry
-- no one has to type it in. This keeps a robust running estimate per ID,
used wherever a size is still needed: placing a tag that only ONE camera
sees (its distance comes from its apparent size), and the pose tab's
per-tag distances. Sizes are in the world calibration's units, so single-
and multi-camera placements always agree.

An ID whose measurements keep disagreeing (e.g. two physical tags of
different sizes sharing it) is flagged rather than silently averaged.
"""

import threading
import time


class TagSizes:
    MAX_N = 200          # cap on the effective sample count: still adapts slowly
    TOL = 0.12           # a measurement this far (relative) from the estimate is an outlier

    def __init__(self, load, save):
        self._load, self._save = load, save
        self.lock = threading.Lock()
        self.sizes = {}          # str(id) -> {"mm", "n", "rejected", "t"}
        self.dirty, self.saved_t = False, 0.0
        self._loaded = False

    def _ensure(self):
        """Load lazily: the host's config may not be readable yet at import."""
        if not self._loaded:
            self._loaded = True
            try:
                self.sizes = dict(self._load() or {})
            except Exception:
                self.sizes = {}

    def get(self, tid):
        """Learned size (mm) for a tag id, or None if not learned yet."""
        self._ensure()
        e = self.sizes.get(str(int(tid)))
        return e["mm"] if e and e["n"] >= 3 else None

    def observe(self, tid, mm):
        """One triangulated measurement of a tag's side length."""
        if not mm or mm <= 0:
            return
        k = str(int(tid))
        self._ensure()
        with self.lock:
            e = self.sizes.get(k)
            if e is None:
                self.sizes[k] = {"mm": float(mm), "n": 1, "rejected": 0, "t": time.time()}
            elif e["n"] < 5 or abs(mm - e["mm"]) <= self.TOL * e["mm"]:
                n = min(e["n"], self.MAX_N)
                e["mm"] = (e["mm"] * n + mm) / (n + 1)
                e["n"] = e["n"] + 1
                e["t"] = time.time()
            else:
                e["rejected"] += 1
            self.dirty = True
            if time.time() - self.saved_t > 10:
                self.flush_locked()

    def flush_locked(self):
        if self.dirty:
            try:
                self._save({k: {**v, "mm": round(v["mm"], 3)} for k, v in self.sizes.items()})
            except Exception:
                pass
            self.dirty, self.saved_t = False, time.time()

    def flush(self):
        with self.lock:
            self.flush_locked()

    def forget(self, tid):
        self._ensure()
        with self.lock:
            self.sizes.pop(str(int(tid)), None)
            self.dirty = True
            self.flush_locked()

    def listing(self):
        self._ensure()
        with self.lock:
            out = []
            for k, v in sorted(self.sizes.items(), key=lambda kv: int(kv[0])):
                n, rej = v["n"], v.get("rejected", 0)
                out.append({"id": int(k), "mm": round(v["mm"], 2), "n": n,
                            "rejected": rej, "ready": n >= 3,
                            # many disagreeing measurements: two tags share this id?
                            "conflict": rej > 20 and rej > 0.3 * n})
            return out
