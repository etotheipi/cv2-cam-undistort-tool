"""Expose a Luxonis OAK device as a plain UVC webcam (/dev/video*).

OAK cameras are not UVC devices: they enumerate as a Movidius VPU and
only produce video once a host boots a pipeline onto them. This script
boots a minimal RGB->UVC pipeline so the device re-enumerates as a
standard webcam that the calibration bridge (and anything else V4L2)
can use. Keep it running for as long as the camera is needed.

Usage:  python local/oak_uvc.py     (depthai >= 3; Ctrl-C to stop)
"""

import time

import depthai as dai


def main():
    print("booting OAK into UVC mode (Ctrl-C to stop)…")
    # UVC must be declared in the board config BEFORE boot so the device
    # enumerates with a webcam USB descriptor
    config = dai.Device.Config()
    uvc_cfg = dai.BoardConfig.UVC(1920, 1080)
    uvc_cfg.frameType = dai.ImgFrame.Type.NV12
    uvc_cfg.cameraName = "OAK UVC Camera"
    config.board.uvc = uvc_cfg
    device = dai.Device(config)
    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A)
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
