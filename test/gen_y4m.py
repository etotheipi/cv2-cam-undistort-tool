"""Generate synthetic checkerboard videos for Chrome's fake-camera testing.

Usage: python test/gen_y4m.py [output_dir]
Then:  google-chrome --use-fake-device-for-media-stream \
         --use-file-for-fake-video-capture=<dir>/board.y4m ...

board.y4m       15 varied in-frame poses (calibration flows), fx=600 truth
board_tilt.y4m  one static steep pose (measurement/ortho flows)
"""
import sys

import cv2
import numpy as np

W, H, FX, SQUARE, PATTERN = 640, 480, 600.0, 25.0, (9, 6)
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])
px_per_sq = 60
cols_sq, rows_sq = PATTERN[0] + 1, PATTERN[1] + 1
margin = px_per_sq
bw, bh = cols_sq * px_per_sq + 2 * margin, rows_sq * px_per_sq + 2 * margin
board = np.full((bh, bw), 255, np.uint8)
for r in range(rows_sq):
    for c in range(cols_sq):
        if (r + c) % 2 == 0:
            board[margin + r * px_per_sq:margin + (r + 1) * px_per_sq,
                  margin + c * px_per_sq:margin + (c + 1) * px_per_sq] = 0
mm_per_px = SQUARE / px_per_sq
tex_px = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], np.float32)
tex_mm = (tex_px - (margin + px_per_sq)) * mm_per_px


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


def main(outdir="."):
    rng = np.random.RandomState(11)
    poses = []
    while len(poses) < 15:
        rvec = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45),
                         rng.uniform(-0.4, 0.4)])
        tvec = np.array([rng.uniform(-40, 15), rng.uniform(-35, 10),
                         rng.uniform(340, 560)])
        obj = np.hstack([tex_mm, np.zeros((4, 1))]).astype(np.float64)
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        pts = pts.reshape(-1, 2)
        if (pts[:, 0].min() > 8 and pts[:, 0].max() < W - 8 and
                pts[:, 1].min() > 8 and pts[:, 1].max() < H - 8):
            poses.append((rvec, tvec))
    frames = []
    for rv, tv in poses:
        frames += [frame(rv, tv)] * 4
    write(f"{outdir}/board.y4m", frames)
    write(f"{outdir}/board_tilt.y4m",
          [frame([0.9, 0.05, 0.05], [0, -30, 900])] * 40)
    print(f"wrote {outdir}/board.y4m and {outdir}/board_tilt.y4m "
          f"(true fx={FX} @ {W}x{H}, {SQUARE}mm squares, {PATTERN} corners)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
