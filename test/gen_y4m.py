"""Generate synthetic ChArUco videos for Chrome's fake-camera testing.

Usage: python test/gen_y4m.py [output_dir]
Then:  google-chrome --use-fake-device-for-media-stream \
         --use-file-for-fake-video-capture=<dir>/board.y4m ...

board.y4m       15 varied in-frame poses (calibration flows), fx=1000 truth
board_tilt.y4m  one static steep pose (measurement/ortho/pose flows)

1280x720 so the 36h11 markers have enough pixels to decode (the standard
8x10 board: ~60 px squares -> ~5 px per tag module at typical poses).
"""
import sys

import cv2
import numpy as np

W, H, FX = 1280, 720, 1000.0
SX, SY, SQUARE, MARKER = 8, 10, 24.0, 17.0        # quickstart default board
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
charuco = cv2.aruco.CharucoBoard((SX, SY), SQUARE, MARKER, dictionary)
px_per_sq = 40                                    # texture resolution
margin = px_per_sq
bw, bh = SX * px_per_sq + 2 * margin, SY * px_per_sq + 2 * margin
board = charuco.generateImage((bw, bh), marginSize=margin, borderBits=1)
mm_per_px = SQUARE / px_per_sq
tex_px = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], np.float32)
tex_mm = (tex_px - np.array([bw / 2, bh / 2])) * mm_per_px   # centered


def frame(rvec, tvec):
    obj = np.hstack([tex_mm, np.zeros((4, 1))]).astype(np.float64)
    pts, _ = cv2.projectPoints(obj, np.asarray(rvec, float),
                               np.asarray(tvec, float), K, None)
    Hm = cv2.getPerspectiveTransform(tex_px, pts.reshape(4, 2).astype(np.float32))
    view = cv2.warpPerspective(board, Hm, (W, H), borderValue=170)
    return cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR),
                        cv2.COLOR_BGR2YUV_I420)


def write(path, frames, fps=4):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for fr in frames:
            f.write(b"FRAME\n")
            f.write(fr.tobytes())


def poses(n=15, seed=11):
    rng = np.random.RandomState(seed)
    out = []
    obj = np.hstack([tex_mm, np.zeros((4, 1))]).astype(np.float64)
    while len(out) < n:
        rvec = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45),
                         rng.uniform(-0.4, 0.4)])
        tvec = np.array([rng.uniform(-70, 70), rng.uniform(-40, 40),
                         rng.uniform(650, 1100)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        pts = pts.reshape(-1, 2)
        if (pts[:, 0].min() > 10 and pts[:, 0].max() < W - 10 and
                pts[:, 1].min() > 10 and pts[:, 1].max() < H - 10):
            out.append((rvec, tvec))
    return out


def main(outdir="."):
    frames = []
    for rv, tv in poses():
        frames += [frame(rv, tv)] * 3
    write(f"{outdir}/board.y4m", frames)
    write(f"{outdir}/board_tilt.y4m",
          [frame([0.6, 0.1, 0.05], [0, -30, 650])] * 40)
    print(f"wrote {outdir}/board.y4m and {outdir}/board_tilt.y4m "
          f"(true fx={FX} @ {W}x{H}, charuco {SX}x{SY} "
          f"{SQUARE}mm/{MARKER}mm 36h11)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
