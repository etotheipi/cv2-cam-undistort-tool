"""Expose a Luxonis OAK device as a plain UVC webcam (/dev/video*).

OAK cameras are not UVC devices: they enumerate as a Movidius VPU and
only produce video once a host boots a pipeline onto them. This script
boots a minimal RGB->UVC pipeline so the device re-enumerates as a
standard webcam that the calibration bridge (and anything else V4L2)
can use. Keep it running for as long as the camera is needed.

Usage:  python local/oak_uvc.py [--focus 0..255]     (depthai >= 3)

--focus locks the lens at a fixed position (135 is a reasonable
near-hyperfocal start for long-range work; higher = closer focus).
Lock it for calibration AND deployment: autofocus shifts the focal
length slightly ("focus breathing"), so a locked lens is the standard
way to keep one calibration exactly valid. Without --focus the camera
runs continuous autofocus.
"""

import argparse
import time

import depthai as dai


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--focus", type=int, default=None, metavar="0..255",
                    help="lock manual focus at this lens position")
    args = ap.parse_args()
    print("booting OAK into UVC mode (Ctrl-C to stop)…")
    # UVC must be declared in the board config BEFORE boot so the device
    # enumerates with a webcam USB descriptor
    config = dai.Device.Config()
    uvc_cfg = dai.BoardConfig.UVC(1920, 1080)
    uvc_cfg.frameType = dai.ImgFrame.Type.NV12
    # bake the focus setting into the USB device name: the OAK's UVC mode
    # exposes no V4L2 focus control, so this is how the value stays
    # visible in the UI and lands in calibration files
    uvc_cfg.cameraName = ("OAK UVC F%d" % args.focus
                          if args.focus is not None else "OAK UVC AF")
    config.board.uvc = uvc_cfg
    device = dai.Device(config)
    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A)
        if args.focus is not None:
            cam.initialControl.setManualFocus(max(0, min(255, args.focus)))
            print(f"lens locked at focus position {args.focus}")
        out = cam.requestOutput((1920, 1080), dai.ImgFrame.Type.NV12,
                                fps=30)
        uvc = pipeline.create(dai.node.UVC)
        out.link(uvc.input)
        pipeline.start()
        print("pipeline running — the OAK should appear as /dev/video* "
              "within a few seconds")
        try:
            while pipeline.isRunning():
                time.sleep(5)
        except KeyboardInterrupt:
            print("stopping UVC mode")


if __name__ == "__main__":
    main()
