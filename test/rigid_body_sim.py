"""Can the D12's rigidity be exploited when sticker placement is unknown?

Prototype + Monte-Carlo comparison (no cameras needed). Each trial builds
one physical D12 with imperfect stickers (off-centre, arbitrary spin,
slightly tilted), then:

  1. LEARN it from a short multi-snapshot session: the current per-tag
     solver (Tracker._solve_graph) gives a starting point, then a
     rigid-body bundle adjustment estimates every sticker's placement on
     the body, two ways:
       free    6 unknowns per sticker, no shape assumptions
       faces   3 per sticker (x, y, spin within its face): the faces are
               the planes of a regular dodecahedron of known size
  2. compare each learned model with the true sticker placements;
  3. USE it on new single-snapshot scenes with a new camera rig: each
     camera's pose comes from ALL the stickers it sees at once (PnP on
     non-coplanar points — no flip ambiguity), then a rigid-body bundle
     adjustment; compared with the current solver on the same data.

    .venv/bin/python test/rigid_body_sim.py --trials 10 --learn-snaps 4
"""
import argparse
import math
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local"))
import pose_solver_sim as sim  # noqa: E402
import tracker as tracker_mod  # noqa: E402

INR = sim.d12_inradius(sim.MARKER)
FACES = sim.d12_face_frames(INR)
NORMALS = sim.d12_normals()
CORNERS = tracker_mod.Tracker._tag_objp(sim.MARKER / 2)       # 4x3, tag frame
inv = tracker_mod.Tracker._inv


def T_of(r, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(r, float))[0]
    T[:3, 3] = t
    return T


def rt_of(T):
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def rot_deg(A, B):
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return math.degrees(math.acos(np.clip(c, -1, 1)))


def face_tag(f, u, v, th):
    """T_body<-tag for a sticker on face f at in-plane (u, v) mm, spin th."""
    S = np.eye(4)
    c, s = math.cos(th), math.sin(th)
    S[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    S[:2, 3] = (u, v)
    return FACES[f] @ S


def robust_mean_pose(Ts):
    """Average of poses that agree with the most central one (a flipped
    single-view tag pose must not drag the mean)."""
    if len(Ts) > 2:
        med = min(Ts, key=lambda A: sum(rot_deg(A, B) for B in Ts))
        Ts = [T for T in Ts if rot_deg(T, med) < 10] or [med]
    U, _S, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def obs_list(snaps):
    out = []
    for si, s in enumerate(snaps):
        for n, tags in s["obs"].items():
            K, d = s["k_of"][n]
            for tid, o in tags.items():
                out.append((n, si, tid, o["pts"].astype(np.float64), K, d))
    return out


def rigid_ba(obs, cams, bodies, tags, model, face_of=None, ref=None,
             fixed_tags=False, delta=2.0, max_it=60):
    """Rigid-body bundle adjustment with a Huber loss (delta px).

    cams {n: T_world<-cam}, bodies {s: T_world<-body}, tags {j: T_body<-tag}.
    Unknowns: every camera; every body pose except the first snapshot's
    (gauge); sticker placements unless fixed_tags — 'free': 6 per sticker
    with sticker `ref` fixed (it defines the body frame); 'faces': (u, v,
    spin) per sticker on face face_of[j] (the face geometry defines the
    body frame). Returns (cams, bodies, tags, rms_px)."""
    obs = [o for o in obs if o[2] in tags and o[0] in cams and o[1] in bodies]
    cam_ids, snaps = sorted(cams), sorted(bodies)
    blocks = [("cam", n, 6) for n in cam_ids] + [("body", s, 6) for s in snaps[1:]]
    if not fixed_tags:
        blocks += [("tag", j, 6 if model == "free" else 3)
                   for j in sorted(tags) if not (model == "free" and j == ref)]
    off, idx, o = [], {}, 0
    for b, (kind, key, size) in enumerate(blocks):
        off.append(o)
        idx[(kind, key)] = b
        o += size
    x = np.zeros(o)
    for b, (kind, key, size) in enumerate(blocks):
        if kind == "cam":
            x[off[b]:off[b] + 6] = np.r_[rt_of(inv(cams[key]))]
        elif kind == "body":
            x[off[b]:off[b] + 6] = np.r_[rt_of(bodies[key])]
        elif size == 6:
            x[off[b]:off[b] + 6] = np.r_[rt_of(tags[key])]
        else:                           # project onto the face: (u, v, spin)
            S = inv(FACES[face_of[key]]) @ tags[key]
            x[off[b]:off[b] + 3] = (S[0, 3], S[1, 3], math.atan2(S[1, 0], S[0, 0]))

    def pose(kind, key, xv):
        b = idx.get((kind, key))
        if b is None:
            return bodies[key] if kind == "body" else tags[key]
        p = xv[off[b]:off[b] + blocks[b][2]]
        if kind == "tag" and blocks[b][2] == 3:
            return face_tag(face_of[key], *p)
        return T_of(p[:3], p[3:])        # cam: T_cam<-world

    touching = [[] for _ in blocks]
    for i, (n, s, j, *_rest) in enumerate(obs):
        for key in (("cam", n), ("body", s), ("tag", j)):
            if key in idx:
                touching[idx[key]].append(i)

    def res_obs(i, xv):
        n, s, j, pts, K, d = obs[i]
        T = pose("cam", n, xv) @ pose("body", s, xv) @ pose("tag", j, xv)
        if T[2, 3] <= 0:
            return np.full(8, 1e3)
        p, _ = cv2.projectPoints(CORNERS, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, d)
        return (p.reshape(-1, 2) - pts).ravel()

    def residuals(xv):
        return np.concatenate([res_obs(i, xv) for i in range(len(obs))])

    def jac(xv, r):
        J = np.zeros((len(r), len(xv)))
        for b, (_kind, _key, size) in enumerate(blocks):
            for k in range(size):
                j = off[b] + k
                angle = (size == 6 and k < 3) or (size == 3 and k == 2)
                eps = 1e-6 if angle else 1e-4
                x2 = xv.copy()
                x2[j] += eps
                for i in touching[b]:
                    J[8 * i:8 * i + 8, j] = (res_obs(i, x2) - r[8 * i:8 * i + 8]) / eps
        return J

    def loss(r):
        e = np.sqrt(np.sum(r.reshape(-1, 2) ** 2, axis=1))
        big = e > delta
        rho = np.where(big, 2 * delta * e - delta ** 2, e ** 2)
        w = np.where(big, delta / np.maximum(e, 1e-12), 1.0)
        return float(rho.sum()), np.repeat(np.sqrt(w), 2)

    r = residuals(x)
    cost, sw = loss(r)
    lam = 1e-3
    for _ in range(max_it):
        Jw = jac(x, r) * sw[:, None]
        A, g = Jw.T @ Jw, Jw.T @ (r * sw)
        Dg = np.diag(np.diag(A) + 1e-9)
        step = None
        for _try in range(8):
            try:
                dx = np.linalg.solve(A + lam * Dg, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r2 = residuals(x + dx)
            c2, sw2 = loss(r2)
            if c2 < cost:
                step = (dx, r2, c2, sw2)
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5
        if step is None:
            break
        dx, r, c2, sw = step
        x = x + dx
        done = cost - c2 <= 1e-6 * cost
        cost = c2
        if done:
            break
    return ({n: inv(pose("cam", n, x)) for n in cam_ids},
            {s: pose("body", s, x) for s in snaps},
            {j: pose("tag", j, x) for j in tags},
            float(np.sqrt(np.mean(r ** 2))))


def init_from_per_tag(res):
    """Body model + poses from the per-tag solver's output; body frame =
    the most-observed sticker's frame."""
    cams = {c["node"]: np.array(c["T"]) for c in res["cameras"]}
    node = {(t["id"], t["snap"]): np.array(t["T"]) for t in res["tags"]}
    count = {}
    for j, _s in node:
        count[j] = count.get(j, 0) + 1
    ref = max(count, key=lambda j: (count[j], -j))
    snaps = sorted({s for _j, s in node})
    bodies = {s: node[(ref, s)] for s in snaps if (ref, s) in node}
    tags = {ref: np.eye(4)}
    for _round in range(3):
        for j in count:
            if j not in tags:
                rel = [inv(bodies[s]) @ node[(j, s)] for s in bodies if (j, s) in node]
                if rel:
                    tags[j] = robust_mean_pose(rel)
        for s in snaps:
            if s not in bodies:
                est = [node[(j, s)] @ inv(tags[j]) for j in tags if (j, s) in node]
                if est:
                    bodies[s] = robust_mean_pose(est)
    return cams, bodies, tags, ref


def to_faces(tags, ref):
    """Assign learned stickers to dodecahedron faces and return
    (face_of, T_nominal<-body, ok) — ok False if the assignment is
    implausible (two stickers on one face, or one > 10 deg off any face).
    Sticker `ref` goes to face 0 and its most
    nearly adjacent sticker to a fixed neighbour of face 0 — any directed
    edge is equivalent under the D12's rotation symmetry — then every
    sticker takes the nearest face normal; the body centre is where the
    stickers' planes sit INR from."""
    nrm = {j: T[:3, 2] for j, T in tags.items()}
    adj = 1 / math.sqrt(5)                               # cos 63.43 deg
    j2 = min((j for j in tags if j != ref), key=lambda j: abs(nrm[ref] @ nrm[j] - adj))
    dots = NORMALS @ NORMALS[0]
    dots[0] = -2
    f2 = int(np.argmax(dots))
    A = np.stack([nrm[ref], nrm[j2], np.cross(nrm[ref], nrm[j2])])
    B = np.stack([NORMALS[0], NORMALS[f2], np.cross(NORMALS[0], NORMALS[f2])])
    U, _S, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    face_of = {j: int(np.argmax(NORMALS @ (R @ nrm[j]))) for j in tags}
    misfit = max(math.degrees(math.acos(np.clip(NORMALS[face_of[j]] @ (R @ nrm[j]), -1, 1)))
                 for j in tags)
    ok = len(set(face_of.values())) == len(face_of) and misfit < 10.0
    # centre c (body frame): n_j . (p_j - c) = INR for every sticker
    M = np.stack([nrm[j] for j in tags])
    b = np.array([nrm[j] @ tags[j][:3, 3] - INR for j in tags])
    c = np.linalg.lstsq(M, b, rcond=None)[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ c
    return face_of, T, ok


def solve_with_model(snaps, tags_model, delta=2.0):
    """Camera poses from a known (learned) D12: per camera per snapshot,
    PnP on every visible sticker's corners in the body frame, placed with
    body 0 as the world, then rigid bundle adjustment (stickers fixed)."""
    rel = {}
    for si, s in enumerate(snaps):
        for n, tv in s["obs"].items():
            use = [j for j in tv if j in tags_model]
            if not use:
                continue
            K, d = s["k_of"][n]
            if len(use) == 1:           # one sticker: planar, ambiguous
                j = use[0]
                rel[(n, si)] = tv[j]["T"] @ inv(tags_model[j])
                continue
            P3 = np.concatenate([(tags_model[j][:3, :3] @ CORNERS.T).T + tags_model[j][:3, 3]
                                 for j in use])
            P2 = np.concatenate([tv[j]["pts"] for j in use]).astype(np.float64)
            ok, rv, tv_, inl = cv2.solvePnPRansac(P3, P2, K, d, reprojectionError=4.0,
                                                  iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inl is None or len(inl) < 6:
                ok, rv, tv_ = cv2.solvePnP(P3, P2, K, d, flags=cv2.SOLVEPNP_SQPNP)
                inl = np.arange(len(P3))
            rv, tv_ = cv2.solvePnPRefineLM(P3[inl.ravel()], P2[inl.ravel()], K, d, rv, tv_)
            rel[(n, si)] = T_of(rv.ravel(), tv_.ravel())       # T_cam<-body
    cams, bodies = {}, {0: np.eye(4)}
    for _round in range(len(snaps) + 2):
        for (n, si), T_cb in rel.items():
            if si in bodies and n not in cams:
                cams[n] = bodies[si] @ inv(T_cb)
            elif n in cams and si not in bodies:
                bodies[si] = cams[n] @ T_cb
    cams, _b, _t, rms = rigid_ba(obs_list(snaps), cams, bodies, tags_model, "free",
                                 fixed_tags=True, delta=delta)
    return cams, rms


def layout_errors(est, cams_true):
    """Per-camera centre error after the best rigid fit to the truth."""
    ns = sorted(est)
    E = np.array([est[n][:3, 3] for n in ns])
    Tt = np.array([cams_true[n]["T_wc"][:3, 3] for n in ns])
    U, _S, Vt = np.linalg.svd((E - E.mean(0)).T @ (Tt - Tt.mean(0)))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return dict(zip(ns, np.linalg.norm((E - E.mean(0)) @ R.T - (Tt - Tt.mean(0)), axis=1)))


def model_errors(tags, true_tags, ref):
    """Learned vs true sticker placement, relative to sticker `ref`."""
    pe, re = [], []
    for j, T in tags.items():
        if j == ref:
            continue
        A = inv(tags[ref]) @ T
        B = inv(true_tags[ref]) @ true_tags[j]
        pe.append(np.linalg.norm(A[:3, 3] - B[:3, 3]))
        re.append(rot_deg(A, B))
    return np.array(pe), np.array(re)


def most_seen(snaps):
    seen = {}
    for tags in snaps[0]["obs"].values():
        for tid in tags:
            seen[tid] = seen.get(tid, 0) + 1
    return max(seen, key=lambda k: (seen[k], -k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--learn-snaps", type=int, default=4)
    ap.add_argument("--learn-radius", type=float, default=600.0)
    ap.add_argument("--tests", type=int, default=3, help="test scenes per trial and radius")
    ap.add_argument("--sticker-mm", type=float, default=2.0)
    ap.add_argument("--tilt-deg", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tr = tracker_mod.Tracker({})
    noisy = dict(sigma=0.3, bad_cam=1, bad_sigma=1.2, n_outliers=3)
    merr = {"free": ([], []), "faces": ([], [])}
    runs = {r: {k: [] for k in ("per-tag", "free", "faces")} for r in (1200.0, 600.0)}
    rejected = 0
    for t in range(a.trials):
        rng = np.random.default_rng(a.seed + t)
        d12 = sim.make_d12(rng, a.sticker_mm, a.tilt_deg)
        true_tags = sim.d12_body_tags(INR, d12)
        # ---- learning session ----
        camsL = sim.make_rig(rng, a.learn_radius)
        snapsL, _ = sim.simulate(camsL, rng, a.learn_snaps, spread=250 * a.learn_radius / 1200,
                                 obj="d12", d12=d12, **noisy)
        res = tr._solve_graph(snapsL, most_seen(snapsL), sim.MARKER)
        cams0, bodies0, tags0, ref = init_from_per_tag(res)
        obsL = obs_list(snapsL)
        cams_f, bodies_f, tags_free, _rms = rigid_ba(obsL, cams0, bodies0, tags0, "free", ref=ref)
        # snap the FREE result onto the faces: it has already corrected any
        # flipped sticker the per-tag start carried (a flipped normal would
        # land on the wrong face)
        face_of, Tnb, ok = to_faces(tags_free, ref)
        if ok:
            _c, _b, tags_faces, _rms = rigid_ba(
                obsL, cams_f, {s: B @ inv(Tnb) for s, B in bodies_f.items()},
                {j: Tnb @ T for j, T in tags_free.items()}, "faces", face_of=face_of)
        else:
            rejected += 1
            tags_faces = tags_free          # implausible assignment: keep free
        for name, tg in (("free", tags_free), ("faces", tags_faces)):
            pe, re = model_errors(tg, true_tags, ref)
            merr[name][0].extend(pe)
            merr[name][1].extend(re)
        # ---- use it: new rigs, single snapshot ----
        for radius in runs:
            for _k in range(a.tests):
                camsT = sim.make_rig(rng, radius)
                snapsT, _ = sim.simulate(camsT, rng, 1, spread=250 * radius / 1200,
                                         obj="d12", d12=d12, **noisy)
                r1 = tr._solve_graph(snapsT, most_seen(snapsT), sim.MARKER)
                est = {"per-tag": {c["node"]: np.array(c["T"]) for c in r1["cameras"]}}
                for name, tg in (("free", tags_free), ("faces", tags_faces)):
                    est[name] = solve_with_model(snapsT, tg)[0]
                for name, e in est.items():
                    runs[radius][name].extend(layout_errors(e, camsT).values())
        print(f"  trial {t + 1}/{a.trials} done", file=sys.stderr, flush=True)

    print(f"{a.trials} D12s: stickers ~{a.sticker_mm} mm off-centre (sd), arbitrary spin, "
          f"~{a.tilt_deg} deg tilt; learned from {a.learn_snaps} snapshots at "
          f"{a.learn_radius:.0f} mm; noisy camera + 3 outlier corners per scene")
    print(f"\nface assignment rejected (kept free) in {rejected} of {a.trials} trials")
    print("Learned sticker placement vs truth (relative to the reference sticker):")
    for name, (pe, re) in merr.items():
        print(f"  {name:6s} position mean {np.mean(pe):5.2f} mm  max {np.max(pe):5.2f}   "
              f"orientation mean {np.mean(re):5.2f} deg  max {np.max(re):5.2f}")
    print("\nSingle-snapshot camera layout error on NEW rigs (frame-free):")
    for radius, byname in runs.items():
        for name, errs in byname.items():
            e = np.array(errs)
            print(f"  rig {radius:4.0f} mm  {name:8s} mean {e.mean():7.1f} mm  median {np.median(e):6.1f}  "
                  f"failures (>100 mm) {int(np.sum(e > 100)):3d} of {len(e)}")


if __name__ == "__main__":
    main()
