# cv2-cam-undistort-tool

Browser-based webcam intrinsic calibration + checkerboard measurement —
**fully client-side**. Camera capture and UI are JavaScript; every OpenCV
operation (corner detection, `calibrateCamera`, undistortion, plane
measurement) runs in **real Python** (`opencv-python` + numpy) inside the
browser via [Pyodide](https://pyodide.org). No server, no uploads — frames
never leave your machine.

**Live app:** https://etotheipi.github.io/cv2-cam-undistort-tool/

## Usage

1. **Collect & Calibrate** — grant camera access, pick a camera and
   resolution, set your printed checkerboard's square size and inner-corner
   counts. Click *Collect Calibration Images*: 5 s countdown, then a frame
   every 2 s; frames without a detectable board are discarded. Newest
   thumbnails appear first so the live view never scrolls away. Space stops
   collection (or snaps a single frame when idle). 20 images minimum,
   40+ ideal.
2. **Calibration Results** — step checklist, log, per-view reprojection
   errors, detected-vs-reprojected overlays. Download the calibration JSON,
   or just keep it — it's saved in your browser's localStorage per camera.
3. **Measure** — original and undistorted live views. A calibration is
   active automatically if this camera was calibrated here before, or load
   a previously downloaded JSON file. Snap a frame (button or Space), click
   *Measure*, then click two points on the checkerboard plane to get the
   real-world distance. The measurement board's size/corners can differ
   from the calibration board — controls default to the calibration values.
   *Save PNG* downloads the frame with measurement overlays burned in.

Only detected corner coordinates and small thumbnails are persisted
(localStorage), so storage stays tiny; calibration itself only needs the
corners.

## Run locally

Any static file server works (camera access requires `localhost` or HTTPS):

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Pyodide + opencv-python (~15 MB) load from the jsDelivr CDN on first visit
and are cached by the browser afterward, so an internet connection is
needed at least once.

## Deployment

GitHub Pages, "deploy from branch" (`main`, root). Pushing to `main` **is**
the deployment — no Actions, no build step. `.nojekyll` tells Pages to
serve files verbatim.

## Calibration file format

Same schema as the companion local tool
([uvc_camera_cal](https://github.com/etotheipi)): top-level `intrinsic`
(camera matrix, distortion coefficients, RMS, checkerboard spec, capture
metadata) and an `extrinsic` placeholder for later. Files produced by
either tool can be loaded into the Measure tab, or consumed downstream:

```python
import cv2, json, numpy as np
cal = json.load(open("my_camera.json"))["intrinsic"]
K = np.array(cal["camera_matrix"]); d = np.array(cal["dist_coeffs"])
undistorted = cv2.undistort(frame, K, d)
```

Browser caveat vs. the local tool: USB serial numbers and full V4L2 mode
lists aren't exposed to web pages, so camera identity is label + VID:PID
(when the browser includes it) plus your chosen name.
