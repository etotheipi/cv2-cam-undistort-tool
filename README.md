# cv2-cam-undistort-tool

Browser-based multi-camera calibration, measurement and ArUco tag
tracking. Camera capture and UI are JavaScript; the calibration/measure
CV (ChArUco detection, `calibrateCamera`, undistortion, plane
measurement) runs in **real Python** (`opencv-python` + numpy) inside the
browser via [Pyodide](https://pyodide.org). In lab mode, a small Flask
bridge owns the host's cameras and additionally runs the live tag
tracker and multi-camera pose solver natively.

**Live app (browser mode):** https://etotheipi.github.io/cv2-cam-undistort-tool/

Everything is standardized on **ChArUco / AprilTag 36h11**: one
dictionary for the calibration board, the measurement board, and
standalone object tags.

## Two ways to run the same app

- **Browser mode** (the Pages link, or any static server): cameras via
  getUserMedia. Zero install; calibration + measurement only (no
  host-camera tabs), and browsers can't read USB serials or mode lists.
- **Local (lab) mode** — the full tool:

  ```bash
  pip install -r local/requirements.txt
  python local/server.py --port 8124            # then open http://localhost:8124
  ```

  The bridge enumerates and owns the host's cameras (Linux/V4L2): real
  serial numbers, full mode lists, many cameras at once. The page
  auto-detects the bridge and unlocks the host-mode tabs. Cameras
  without unique serials (clone hardware often ships `SN001`, `UC852`
  etc.) are flagged — assign each a short label (write it on the camera
  body); it becomes part of the calibration identity.

### Setting up a new machine

1. Clone; `pip install -r local/requirements.txt` (any Python ≥3.10;
   OpenCV ≥4.8 — the aruco module is required).
2. **Copy `.env` manually** — it holds the S3 credentials and is
   deliberately gitignored. (Or press *Reveal S3 credentials* in the
   Cameras tab's storage panel on the old machine and paste.) Without
   it, storage falls back to a local `camera_cal/` directory.
3. `local/config.json` (storage selection) is auto-created; no transfer
   needed.
4. **Multiple cameras per USB controller**: every USB 2 webcam on one
   controller (hubs included) shares one ~4,800-byte-per-125 µs
   bandwidth budget, and cameras reserve a fixed worst-case slice of it
   (e.g. an Arducam OV9782 reserves 2,400 at 640×480 but 3,072 at
   720p). On strict controllers that means **one camera per controller
   at 720p** — spread cameras across controllers (the USB Topology panel
   on the Cameras tab shows the live tree per controller).
   `options uvcvideo quirks=128` is often suggested for this, but the
   driver only applies it to uncompressed formats; this tool streams
   MJPEG, so it doesn't help. The optional
   [uvc-mjpeg-bandwidth extra](extras/uvc-mjpeg-bandwidth/) patches the
   driver to cap MJPEG reservations — measured: six cameras at 720p on
   three ports of one card — and explains the trade-offs.
5. **Let the bridge USB-reset cameras** (for the header's *⟲ Reset all*
   button):

   ```bash
   echo 'SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ENV{ID_USB_INTERFACES}=="*:0e01??:*", MODE="0664", GROUP="plugdev"' \
     | sudo tee /etc/udev/rules.d/70-camcal-usb-reset.rules
   sudo udevadm control --reload && sudo udevadm trigger -s usb
   ```

   Your user must be in `plugdev` (`groups` to check; Ubuntu/Mint
   desktop users are by default). This makes every USB device with a UVC
   camera interface (UVC 1.0–1.5, any vendor, including ones plugged in
   later) group-writable, so the bridge can send the port-reset ioctl
   without root. It grants raw USB access to the *whole* device — a
   webcam's built-in mic, or a monitor/dock with a camera, included.
   Verify: `ls -l /dev/bus/usb/*/*` shows cameras as `root plugdev`.
   Without it, *Reset all* still stops every stream and then shows this
   command. The button never resets hubs or whole buses (that would drop
   the keyboard and mouse) and skips Luxonis OAKs (a reset drops them
   out of UVC mode — restart `oak_uvc.py` instead).
6. Pyodide + opencv-python (~15 MB) load from the jsDelivr CDN on first
   visit, then cache — internet needed at least once per browser.

## Tabs (lab mode)

1. **Cameras** (home) — live view of every attached camera with
   per-camera rotate; calibration assignment per camera with
   status-colored actions (red *Calibrate* = missing, yellow = two
   cameras share one file, gray = done); the Calibration Files panel
   (rename / delete / copy between S3 and local dir); and the live USB
   topology tree grouped by controller for balancing bandwidth.
2. **ChArUco Tags & Calibration** — print the quickstart calibration
   board (exact physical scale; PNG + machine-readable manifest), and
   build sheets of standalone object tags (IDs auto-increment; keep the
   quiet zone when cutting).
3. **Collect & Calibrate** — pick camera + resolution, confirm the
   board parameters, collect (5 s countdown, one frame per 2 s, frames
   without a detected board or byte-identical to a previous frame are
   discarded; partial board views count). 20 images minimum, 40+ ideal.
   Or **screen cal**: aim the camera at this monitor and the page flashes
   boards full-screen and captures them itself — *1 position* (straight
   on; solid distortion, unreliable focal length) or *3 positions*
   (straight, turned, tilted — the tilts pin down focal length).
4. **Calibration Results** — pipeline log, per-view reprojection
   errors, reprojection overlays, prune-and-rerun. Calibrations save to
   the storage backend **and** a browser copy automatically.
5. **Measure** — original/undistorted live views; snap a frame to
   measure point-to-point distances on the board plane, view the
   camera↔board pose (solvePnP), or generate metrically-square
   orthogonal views at several zoom levels.
6. **Tracking** — all cameras at once with live ArUco detections
   (ID + distance per tag), independent view/tracker FPS, per-camera
   resolution, and a Load panel (tracker rate/duty, CPU, memory,
   per-camera capture fps). Detection runs natively on the bridge.
   Below: the **World Coordinate System** solver — one-shot, or
   multi-snapshot *World Calibration* (move a reference block between
   snapshots; joint bundle adjustment over all observations; world
   frame = tag 555). Interactive 3D result + annotated per-camera
   frames.

## Calibration storage (lab mode)

Configured in the Cameras tab (or `--storage dir:/path` / `--storage s3`
/ `--env-file`):

- **Local directory** (default `camera_cal/`): one JSON per camera slug.
- **S3**: files live at
  `s3://multi-cam-calibration-files-{account_id}/{iam_username}/{slug}.json`.
  Bucket derives from the AWS account and prefix from the caller's IAM
  username (STS), so the credentials alone determine the location.
  Credential precedence: process env vars → `.env` (never overrides) →
  `~/.aws/credentials`. Cycling access keys keeps the prefix intact.
- The ⧉ button on any calibration copies it to the other backend.

One-time AWS setup per account + deployment (needs admin credentials):

```bash
python local/provision_aws.py camcal-01 --env-file .env
```

creates the bucket (public access blocked), a managed policy scoped by
`${aws:username}` to each user's own prefix, the deployment user, and a
permanent access key (shown once; optionally written to `.env`).

## Calibration file format

Top-level `intrinsic` (camera matrix, distortion coefficients, RMS,
intrinsic std deviations, ChArUco board manifest, capture metadata) and
`extrinsic` (currently the view rotation). Consume downstream:

```python
import cv2, json, numpy as np
cal = json.load(open("my_camera.json"))["intrinsic"]
K = np.array(cal["camera_matrix"]); d = np.array(cal["dist_coeffs"])
undistorted = cv2.undistort(frame, K, d)
```

## Deployment (browser mode)

GitHub Pages, "deploy from branch" (`main`, root). Pushing to `main`
**is** the deployment — no Actions, no build step. Development happens
on feature branches (currently `charuco`); merging to `main` deploys.

## Tests

`test/gen_y4m.py` renders synthetic ChArUco videos (known intrinsics)
for Chrome's fake-camera flags, used by the Playwright ground-truth
tests: the full in-browser collect→calibrate flow recovers the true
focal length within ~0.1%, and the tracker/world-solver have native
ground-truth tests for tag distance and multi-snapshot pose chaining.
