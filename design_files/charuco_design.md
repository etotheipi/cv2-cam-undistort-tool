## Recommended path

Use an OpenCV ChArUco board populated from the AprilTag 36h11 dictionary:

- cv::aruco::DICT_APRILTAG_36h11

Then standardize on these APIs:

- CharucoBoard
- CharucoBoard::generateImage()
- CharucoDetector::detectBoard()
- Board::matchImagePoints()
- calibrateCameraExtended()
- solvePnP()
- ArucoDetector::detectMarkers()

That gives you:

Chessboard-style subpixel intersections for accurate intrinsic calibration.
Encoded IDs, orientation, partial-board detection, and occlusion tolerance.
The same dictionary for printable standalone object tags.
An entirely OpenCV-based detection and pose pipeline.

OpenCV explicitly recommends ChArUco corners over ordinary ArUco-board marker corners for calibration because the ChArUco intersections are more accurate. Partial board views are also permitted.

DICT_APRILTAG_36h11 contains 587 IDs with a minimum Hamming distance of 11, which is a good balance for a system needing many object IDs. OpenCV exposes it as a predefined dictionary, so it can be passed directly to both CharucoBoard and ArucoDetector.

### Why not use an AprilGrid directly?

An AprilGrid, as used by tools such as Kalibr, is good, but it does not give you the chessboard intersections that make ChArUco attractive for intrinsic calibration. You would also be maintaining more calibration-specific code outside OpenCV.

Upstream AprilTag currently recommends tagStandard41h12 for most new AprilTag-only applications. OpenCV’s predefined AprilTag choices, however, are the classic 16h5, 25h9, 36h10, and 36h11 families. Therefore, 36h11 is the practical choice when OpenCV-only standardization and one shared family matter more than using the newest upstream family.

## Board design

A good first board would be:

- Dictionary:       DICT_APRILTAG_36h11
- Squares:          8 × 11 or 7 × 10
- Square length:    30–40 mm
- Marker length:    approximately 65–75% of square length
- Border bits:      1
- Board material:   rigid, flat, matte

Use the largest board that is practical in your camera working volume. A3 or A2 is considerably better than letter/A4 for cameras that will operate several meters away.

OpenCV can generate the board directly with CharucoBoard::generateImage(), and current OpenCV also includes apps/pattern-tools/generate_pattern.py. Lengths can be expressed in any consistent unit; pose translations will come back in that same unit.

For example:

```python
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

board = cv2.aruco.CharucoBoard(
    (8, 11),       # squares X, Y
    0.035,         # 35 mm squares, expressed in meters
    0.025,         # 25 mm markers
    dictionary
)

image = board.generateImage(
    (2400, 3300),
    marginSize=100,
    borderBits=1
)

cv2.imwrite("charuco_apriltag36h11.png", image)
```

Print at exactly 100%, disable “fit to page,” and measure several squares after printing. Printer scaling and a warped backing can easily become larger error sources than the corner detector.

Keep a machine-readable board manifest beside the image:

```
type: charuco
dictionary: DICT_APRILTAG_36h11
squares_x: 8
squares_y: 11
square_length_m: 0.035
marker_length_m: 0.025
border_bits: 1
marker_ids: [0, 1, 2, ...]
legacy_pattern: false
```

Persisting legacy_pattern matters because OpenCV changed the ChArUco pattern convention after 4.6 for boards with even row counts. Current OpenCV provides setLegacyPattern() for reading older printed boards.

1. Intrinsic calibration workflow

Perform this separately for every physical camera and every operating configuration.

Changing resolution, sensor crop, focus, zoom, stabilization, or lens position can invalidate calibration. Lock autofocus and zoom before collecting data.

Capture

Collect roughly 25–50 accepted images per camera, with deliberate diversity:

- Board near every image edge and corner.
- Several distances.
- Significant pitch and yaw, not just front-facing images.
- Some partial views.
- Sharp images without motion blur.
- Board covering both large and moderate fractions of the image.

More images are not automatically better. Thirty geometrically diverse images are usually more useful than hundreds of nearly identical front-facing views.

Detect and construct correspondences

Use the current detector API:

```python
charuco_params = cv2.aruco.CharucoParameters()
detector_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.CharucoDetector(
    board,
    charuco_params,
    detector_params
)

all_object_points = []
all_image_points = []

for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < 8:
        continue

    object_points, image_points = board.matchImagePoints(
        charuco_corners,
        charuco_ids
    )

    all_object_points.append(object_points)
    all_image_points.append(image_points)
```

CharucoDetector::detectBoard() replaces older combinations such as detectMarkers() plus interpolateCornersCharuco(). matchImagePoints() converts the varying visible corner IDs in each frame into correctly matched 3D/2D point arrays.

### Calibrate

Use the general calibration API rather than building new code around the older specialized calibrateCameraCharuco() helper:

```python
result = cv2.calibrateCameraExtended(
    all_object_points,
    all_image_points,
    image_size,
    None,
    None,
    flags=0
)

rms = result[0]
camera_matrix = result[1]
dist_coeffs = result[2]
rvecs = result[3]
tvecs = result[4]
std_intrinsics = result[5]
std_extrinsics = result[6]
per_view_errors = result[7]
```

Current OpenCV examples use matchImagePoints() followed by the generic calibrateCamera() family, and several older ArUco-specific calibration and pose helpers are deprecated in favor of point matching and solvePnP().

Start with the normal five-coefficient model:

- k1, k2, p1, p2, k3

Do not immediately enable rational, thin-prism, or tilted-sensor terms. Extra coefficients can absorb poor image coverage and produce a deceptively low calibration error. Add them only when residual plots or held-out validation justify them. For true fisheye lenses, feed the same matched ChArUco points into OpenCV’s fisheye calibration model instead.

Reject and repeat

Inspect:

- Per-view reprojection error.
- Intrinsic standard deviations.
- Projected-versus-detected residual vectors.
- Undistortion near all four image corners.
- Held-out board images not used during calibration.

Remove clearly blurred or anomalous captures and calibrate again. Do not optimize solely for the lowest aggregate RMS.

Save at least:

- camera_id:
- image_width:
- image_height:
- camera_model: opencv_standard
- camera_matrix:
- distortion_coefficients:
- calibration_rms:
- intrinsic_std_deviations:
- per_view_errors:
- board_manifest_hash:
- opencv_version:


2. Static extrinsic calibration

The cleanest approach is to let the ChArUco board define your world coordinate system.

- Simple case: every camera can see one fixed board pose
- Install all cameras and load each camera’s intrinsic calibration.
- Place the board rigidly at the desired world origin.
- Do not move it during the extrinsic procedure.
- Capture several sharp images from each camera.
- Detect ChArUco corners.
- Match board and image points.
- Run solvePnP() for each camera.
- Average or jointly refine the repeated estimates.

```python
object_points, image_points = board.matchImagePoints(
    charuco_corners,
    charuco_ids
)

ok, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
```

OpenCV defines the result as the transform from object or world coordinates into camera coordinates:

`X_camera = R_camera_world · X_world + t_camera_world`

Therefore:

- T_camera_world = [R | t]
- T_world_camera = inverse(T_camera_world)

The latter is normally what you want when storing the camera’s physical pose in the environment.

For camera i and camera j:

T_camera_j_camera_i =
    T_camera_j_world · inverse(T_camera_i_world)

Use more than the minimum four corners. A large ChArUco board with dozens of visible intersections is much more stable than estimating the camera extrinsic from one small standalone marker.

### Important limitation

You cannot move the board to an arbitrary unrelated pose for each camera and still obtain mutually consistent camera extrinsics. Each solvePnP() result would then be relative to a different board frame.

When every camera cannot see the same fixed board, use one of these:

- Several fixed boards whose transforms in the world are measured.
- A larger portable rigid target placed in one surveyed world pose at a time.
- Simultaneous observations where two or more cameras see the moving board at the same instant.
- An overlapping camera graph followed by global bundle adjustment.

For simultaneous camera observations of the same board pose k:

T_camera_j_camera_i = T_camera_j_board_k · inverse(T_camera_i_board_k)

Collect many shared poses and optimize one fixed transform per camera. Do not simply average Rodrigues vectors and translations independently; optimize transforms using reprojection error.

3. Individual object tags

Reserve non-overlapping ID ranges:

- Board markers:        IDs 0–49
- Tracked objects:      IDs 100–499
- Infrastructure tags:  IDs 500–586

The exact ranges depend on how many markers your board consumes.

Generate individual tags with the same dictionary:

```python
tag = cv2.aruco.generateImageMarker(
    dictionary,
    object_tag_id,
    1000,
    borderBits=1
)

cv2.imwrite(f"tag_{object_tag_id:03d}.png", tag)
```

At runtime:

```python
detector_params = cv2.aruco.DetectorParameters()
tag_detector = cv2.aruco.ArucoDetector(
    dictionary,
    detector_params
)

corners, ids, rejected = tag_detector.detectMarkers(gray)
```

OpenCV supports individual marker detection and then uses solvePnP() for pose estimation. For exactly one square marker, SOLVEPNP_IPPE_SQUARE is specifically intended for square-marker pose estimation.

For each object, store:

- tag_id:
- tag_side_length_m:
- T_object_tag:

Then:

T_camera_object = T_camera_tag · inverse(T_object_tag)

T_world_object = T_world_camera · T_camera_object


For better object pose

A single planar tag is adequate for identification and many pose applications, but its depth and orientation can become noisy when it is small, distant, or almost front-facing.

For valuable objects, use a rigid multi-tag object definition:

Two or more tags.
Different faces or non-coplanar orientations when possible.
Known transform from every tag to the object frame.
One combined PnP solve using all visible corners.

You can represent that as an OpenCV Board, call matchImagePoints(), and solve one pose from all visible tags. OpenCV describes a board as a set of markers acting as one object for pose estimation.

Detection configuration

Begin with default DetectorParameters. Add configuration only after collecting difficult examples.

Useful adjustments commonly include:

```
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.minMarkerPerimeterRate = ...
params.maxMarkerPerimeterRate = ...
params.adaptiveThreshWinSizeMin = ...
params.adaptiveThreshWinSizeMax = ...
```

For ChArUco calibration, the final calibration points are the chessboard intersections, so concentrate on getting reliable marker decoding and good ChArUco interpolation rather than over-tuning raw tag corner refinement.

Generate all boards and individual tags with the same generator/version, and include automatic tests that verify:

- Printed ID.
- Returned ID.
- Corner order.
- Rotation convention.
- Physical side-length convention.
- Projection of the recovered pose back onto the image.

That prevents interoperability surprises if you later mix OpenCV detection with the native AprilTag library.

## Licensing

This stack is suitable for an open-source or commercial product:

OpenCV 4.5 and later is Apache 2.0 licensed and explicitly permits commercial use.
The upstream AprilTag implementation is BSD 2-Clause licensed.
The upstream repository of generated AprilTag images and SVG conversion utility is also BSD 2-Clause licensed.

Include the required OpenCV and AprilTag notices in your distribution. This is a permissive setup; it does not require your application itself to use a copyleft license.

Final architecture

```
Board/tag definition
    └── DICT_APRILTAG_36h11
        ├── One rigid ChArUco calibration board
        └── Standalone object tags / multi-tag object boards

Per-camera intrinsic calibration
    └── CharucoDetector
        └── matchImagePoints
            └── calibrateCameraExtended
                └── K, distortion, diagnostics

Installed-camera extrinsic calibration
    └── Fixed ChArUco board defining world coordinates
        └── matchImagePoints
            └── solvePnP
                └── T_camera_world and T_world_camera

Runtime tracking
    └── ArucoDetector using the same dictionary
        └── solvePnP / multi-marker Board PnP
            └── T_camera_object
                └── T_world_object
```

The central design decision is: use ChArUco for calibration accuracy, use the AprilTag 36h11 code family for IDs, and use generic OpenCV point-matching plus PnP APIs for all pose calculations.