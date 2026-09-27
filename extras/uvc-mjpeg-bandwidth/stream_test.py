#!/usr/bin/env python3
"""Stream several cameras at once and check that every one really works.

    python3 stream_test.py 1280x720 11-1.1 11-1.2 11-1.3     # USB port paths
    N=300 python3 stream_test.py 1280x720 7-1 13-1.3 ...     # more frames

Per camera: the alternate setting (bandwidth) it reserved, frames captured,
frames that decode as complete JPEGs, and the fps v4l2-ctl measured. Then
any bandwidth / capping messages the kernel logged meanwhile. Port paths
("11-1.2") are what `usb_reservations.py` and `lsusb -t` show. Cameras must
be free (stop the bridge server first). Needs v4l2-ctl (v4l-utils) and
python3 with numpy + opencv (the repo's .venv has both).

Notes from testing: "fps" drops to ~15 in dim light on cameras with
exposure_dynamic_framerate=1 — that's exposure, not bandwidth (turn it off:
v4l2-ctl -d /dev/videoN -c exposure_dynamic_framerate=0). Some cameras pad
the last packet, so a frame's JPEG end marker can sit a few bytes before
the end of the buffer; that frame is still complete.
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from usb_reservations import SYS, packet_sizes, read


def capture_node(port):
    for v in (SYS / f"{port}:1.0" / "video4linux").iterdir():
        if read(v / "index") == "0":
            return f"/dev/{v.name}"
    raise SystemExit(f"{port}: no video capture node")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    w, h = sys.argv[1].split("x")
    ports = sys.argv[2:]
    n = int(os.environ.get("N", 150))
    since = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stream_test_"))
    procs, alts = {}, {p: 0 for p in ports}
    for p in ports:
        f = tmp / f"{p}.mjpg"
        procs[p] = (subprocess.Popen(
            ["v4l2-ctl", "-d", capture_node(p),
             f"--set-fmt-video=width={w},height={h},pixelformat=MJPG",
             "--stream-mmap", f"--stream-count={n}", f"--stream-to={f}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), f)
    t0 = time.time()
    # poll the active altsetting while streaming: one read can miss it
    while (any(pr.poll() is None for pr, _ in procs.values())
           and time.time() - t0 < n / 5 + 15):
        for p in ports:
            alts[p] = max(alts[p], int(read(SYS / f"{p}:1.1" / "bAlternateSetting") or 0))
        time.sleep(0.05)
    total = 0
    for p, (pr, f) in procs.items():
        if pr.poll() is None:
            pr.kill()
            out = "TIMEOUT"
        else:
            out = pr.stdout.read()
        fps = re.findall(r"([\d.]+) fps", out)
        data = f.read_bytes() if f.exists() else b""
        frames = [b"\xff\xd8" + x for x in data.split(b"\xff\xd8")[1:]]
        good = sum(1 for fr in frames
                   if b"\xff\xd9" in fr[-32:]
                   and cv2.imdecode(np.frombuffer(fr, np.uint8), cv2.IMREAD_GRAYSCALE) is not None)
        size = packet_sizes(SYS / p).get(alts[p], 0)
        total += size
        tail = out.strip().splitlines()[-1][:50] if out.strip() else "no output"
        print(f"  {p:8s} alt {alts[p]:2d} = {size:4d} B  frames {len(frames):3d}/{n}  "
              f"complete {good:3d}  {fps[-1] + ' fps' if fps else tail}")
        f.unlink(missing_ok=True)
    tmp.rmdir()
    print(f"  total reserved across these cameras: {total} B per 125 µs")
    log = subprocess.run(["journalctl", "-k", "--since", since],
                         capture_output=True, text=True).stdout
    for line in log.splitlines():
        if re.search(r"bandwidth|Capping", line, re.I):
            print("  kernel:", line.split("kernel: ")[-1][:110])


if __name__ == "__main__":
    main()
