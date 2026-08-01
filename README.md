# cv2-cam-undistort-tool

Browser-based webcam intrinsic calibration + checkerboard measurement —
**fully client-side**. Camera capture and UI are JavaScript; every OpenCV
operation (corner detection, `calibrateCamera`, undistortion, plane
measurement) runs in **real Python** (`opencv-python` + numpy) inside the
browser via [Pyodide](https://pyodide.org). No server, no uploads — frames
never leave your machine.

**Live app:** https://etotheipi.github.io/cv2-cam-undistort-tool/

## Two ways to run the same app

- **Browser mode** (the Pages link above, or any static server): cameras
  via getUserMedia. Zero install, but browsers can't read USB serial
  numbers or full mode lists.
- **Local (lab) mode**: `pip install -r local/requirements.txt`, then
  `python local/server.py [--port 8123]` from the repo root. A small
  bridge server enumerates and owns the host's cameras directly
  (Linux/V4L2): real serial numbers, stable by-id device IDs, full mode
  lists — and no camera contention with other browser tabs. The page
  auto-detects the bridge; the CV pipeline (Pyodide) is identical in both
  modes. Cameras without unique serials (clone hardware often ships
  `SN001` etc.) are flagged — enter a short label (and write it on the
  camera body); it becomes part of the calibration ID.

## Calibration storage (local mode)

Calibrations save/load automatically through a configurable backend
(Collect tab → Calibration Storage, or `--storage dir:/path` / `--storage
s3` / `--env-file`):

- **Local directory** (default `camera_cal/`): one JSON per camera slug.
- **S3**: files live at
  `s3://multi-cam-calibration-files-{account_id}/{iam_username}/{slug}.json`.
  The bucket comes from the AWS account and the prefix from the caller's
  IAM username (STS), so the credentials alone determine the location —
  no separate path config, and multiple deployments in one account can't
  collide. Credential precedence: process env vars → `.env` file (never
  overrides) → `~/.aws/credentials`. Cycling a user's access keys keeps
  the prefix and files intact.

One-time setup per AWS account + deployment (needs admin credentials):

```bash
python local/provision_aws.py camcal-01 --env-file .env
```

creates the bucket (public access blocked), a shared managed policy
scoped by `${aws:username}` to each user's own prefix, the deployment
user, and a permanent access key — shown once and optionally written to
`.env`. The *Reveal S3 credentials* button in the storage panel shows the
resolved key/secret for copying onto the target (operational) system.

## Usage

1. **Collect & Calibrate** — grant camera access, pick a camera and
   resolution, set your printed checkerboard's square size and inner-corner
   counts. Click *Collect Calibration Images*: 5 s countdown, then a frame
   every 2 s; frames without a detectable board are discarded. Newest
   thumbnails appear first so the live view never scrolls away. Space stops
   collection (or snaps a single frame when idle). 20 images minimum,
   40+ ideal.
2. **Calibration Results** — step checklist, log, per-view reprojection
   errors, detected-vs-reprojected overlays. *Remove Poor Calibration
   Images & Rerun* deletes every image above the error threshold (default
   1.0 px) and recalibrates. Download the calibration JSON, or just keep
   it — it's saved in your browser's localStorage per camera.
3. **Measure** — original and undistorted live views, with camera and
   resolution selectors (the actual delivered mode is always shown; a mode
   the camera can't produce falls back with a warning). A calibration is
   active automatically if this camera was calibrated here before, or load
   a previously downloaded JSON file. Snap a frame (Space snaps the
   undistorted view; each feed also has a button), click *Measure*, then
   click two points on the checkerboard plane to get the real-world
   distance. Measurement-board controls sit below the snapped frame,
   default to the calibration board, and are remembered per camera.
   *Save PNG* downloads the frame with measurement overlays burned in.
   *Generate Orthogonal View* warps the frame so the board plane is
   metrically square, at up to 4 zoom levels (board diagonal = 1/6, 1/18,
   1/50 of the view diagonal, plus a best-effort everything-included view;
   wider levels only appear while the previous one still crops the image).
   Each level is a clickable thumbnail with its own measure/save; a
   measurement carries across zoom levels via board coordinates.

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
