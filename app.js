/* Webcam Calibration & Measure — fully client-side (Pyodide + opencv-python).
   Camera capture and UI in JS; all computer vision in Python via calib_core.py. */
"use strict";

const $ = (id) => document.getElementById(id);
const UNIT_TO_MM = { mm: 1, m: 1000, in: 25.4, ft: 304.8 };

let toastTimer = null;
function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3800);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function slugify(label) {
  return (label || "camera").replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "").slice(0, 80) || "camera";
}

function nowStamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

/* ------------------------------------------------------------------ state */
let HOST = false;        // true when the local bridge server is detected
const S = {
  pyReady: false,
  devices: [],
  device: null,          // selected MediaDeviceInfo (browser mode)
  hostCam: null,         // selected host camera metadata (local mode)
  slug: null,
  stream: null,
  trackSettings: {},
  trackCaps: null,
  micPresent: false,
  collecting: false,
  collectTimers: [],
  images: [],            // records for current camera (from localStorage)
  lastResult: null,      // last calibration result (full JSON object)
  lastRun: null,         // {ids, perView} from the last solve, for pruning
  activeCal: null,       // calibration used by the measure tab
  activeCalSource: "",   // "calibrated" | "storage" | "uploaded file"
  snap: null,            // {imageData, w, h, source}
  snapBoard: null,       // corners from prepare_measure, or null
  measuring: false,
  measurePts: [],
  measurements: [],
};

let py = null;           // Python module proxy (calib_core)

/* ---------------------------------------------------------------- storage */
const LS = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
    catch { return fallback; }
  },
  set(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); return true; }
    catch (e) {
      toast("Browser storage is full — delete some images. " + e.message, true);
      return false;
    }
  },
  del(key) { localStorage.removeItem(key); },
  images: (slug) => LS.get(`cvcal:images:${slug}`, []),
  setImages: (slug, arr) => LS.set(`cvcal:images:${slug}`, arr),
  cal: (slug) => LS.get(`cvcal:calib:${slug}`, null),
  setCal: (slug, obj) => LS.set(`cvcal:calib:${slug}`, obj),
  delCal: (slug) => LS.del(`cvcal:calib:${slug}`),
};

/* Board params are ChArUco squares (not inner corners); sizes always mm. */
function settings() {
  return {
    sx: parseInt($("cols").value, 10),
    sy: parseInt($("rows").value, 10),
    square: parseFloat($("squareSize").value),
    marker: parseFloat($("markerSize").value),
    name: $("camName").value.trim(),
  };
}
function measureSettings() {
  return {
    sx: parseInt($("mCols").value, 10),
    sy: parseInt($("mRows").value, 10),
    square: parseFloat($("mSquareSize").value),
    marker: parseFloat($("mMarkerSize").value),
    units: $("mUnits").value,        // display units only; board is mm
  };
}

function saveForm() {
  LS.set("cvcal:form", {
    sx: $("cols").value, sy: $("rows").value,
    square: $("squareSize").value, marker: $("markerSize").value,
  });
  if (S.slug) LS.set(`cvcal:name:${S.slug}`, $("camName").value);
}
function restoreForm() {
  const f = LS.get("cvcal:form", {});
  if (f.sx) $("cols").value = f.sx;
  if (f.sy) $("rows").value = f.sy;
  if (f.square) $("squareSize").value = f.square;
  if (f.marker) $("markerSize").value = f.marker;
}
["cols", "rows", "squareSize", "markerSize", "camName"].forEach((id) =>
  $(id).addEventListener("change", saveForm));
$("useQuickstart").addEventListener("click", () => {
  const p = chBoardParams();     // the ChArUco tab's current quickstart board
  $("cols").value = p.sx;
  $("rows").value = p.sy;
  $("squareSize").value = p.square;
  $("markerSize").value = p.marker;
  saveForm();
  toast(`Board set to ${p.sx}×${p.sy}, ${p.square} mm squares / ${p.marker} mm markers.`);
});
$("camName").addEventListener("change", () => {
  if (!S.slug) return;
  stopCollecting(true);
  loadImages();           // image sets are per camera identity (label included)
});

/* Measured distances are stored in mm; the display-units dropdown only
   changes how they are shown. */
const fromMm = (mm) => mm / UNIT_TO_MM[$("mUnits").value];

/* ------------------------------------------------------------------- boot */
function setBoot(msg, cls = "") {
  const b = $("bootStatus");
  b.textContent = msg;
  b.className = "boot " + cls;
}

async function bootPython() {
  try {
    setBoot("loading Python runtime…");
    const pyodide = await loadPyodide();
    setBoot("loading OpenCV + numpy (~15 MB, cached after first visit)…");
    await pyodide.loadPackage(["numpy", "opencv-python"]);
    setBoot("loading calibration core…");
    const src = await (await fetch("py/calib_core.py", { cache: "no-cache" })).text();
    pyodide.FS.writeFile("/home/pyodide/calib_core.py", src);
    pyodide.runPython("import calib_core");
    py = pyodide.globals.get("calib_core");
    S.pyReady = true;
    setBoot(`ready — Python + OpenCV ${py.cv2_version()}` +
            (HOST ? " — host cameras" : ""), "ready");
    if (S.activeCal) pushActiveCalToPython();
    updateButtons();
    chPyReady();
  } catch (e) {
    setBoot("failed to load Python: " + e.message, "error");
    toast("Python/OpenCV failed to load — check your connection and reload.", true);
  }
}

/* ------------------------------------------------------------------- tabs */
let activeTab = "collect";
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
function switchTab(name) {
  const prev = activeTab;
  activeTab = name;
  document.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach((p) =>
    p.classList.toggle("active", p.id === "tab-" + name));
  setTimeout(applyOrientationCss, 50);   // fit factor depends on visible layout
  if (name === "cameras") renderConfigTab();
  else if (prev === "cameras") stopGridStreams();
  if (name === "tracking") enterTracking();
  else if (prev === "tracking") leaveTracking();
  if (name === "live") lvEnter();
  else if (prev === "live" && LV.on) lvStop();   // don't hold cameras away
  if (name === "charuco") chRefreshBoard();
}

/* ----------------------------------------------------------- camera setup */
/* Storage identity: base ID from the device, plus an optional user label
   (base__L<label>) for physically-labeled units that share a serial. */
function cleanLabel(v) {
  return (v || "").trim().replace(/[^A-Za-z0-9._-]+/g, "");
}
function storageSlug() {
  const label = cleanLabel($("camName").value);
  return label ? `${S.slug}__L${label}` : S.slug;
}

/* --- calibration family (versions sharing this camera's base ID) --- */
async function loadCalFamily() {
  let all = [];
  try { all = await (await fetch("api/host/calibrations")).json(); } catch {}
  if (!Array.isArray(all)) all = [];
  const fam = [];
  for (const e of all) {
    if (e.slug === S.slug) fam.unshift({ label: null, slug: e.slug });
    else if (e.slug.startsWith(S.slug + "__L")) {
      fam.push({ label: e.slug.slice(S.slug.length + 3), slug: e.slug });
    }
  }
  return fam;
}

function renderCalFamily(fam, selectedLabel) {
  const box = $("calFamily");
  if (!HOST || !fam.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const multi = fam.length > 1 || S.familyExpanded;
  $("calNotMine").classList.toggle("hidden", multi);
  $("calFamilyPick").classList.toggle("hidden", !multi);
  const sel = $("calVersionSel");
  sel.innerHTML = "";
  for (const f of fam) {
    const opt = document.createElement("option");
    opt.value = f.label || "";
    opt.textContent = f.label ? `label ${f.label}` : "default (unlabeled)";
    sel.appendChild(opt);
  }
  sel.value = selectedLabel || "";
}

async function selectCalVersion(label) {
  $("camName").value = label || "";
  if (S.hostCam) rigSet(portKey(S.hostCam), label || "");
  loadImages();          // collected images follow the camera identity
  const ss = storageSlug();
  try {
    const r = await fetch(`api/host/calibrations/${ss}`);
    const cal = r.ok ? await r.json() : null;
    if (cal?.intrinsic?.camera_matrix) {
      LS.setCal(S.slug, cal);
      activateCalibration(cal, "storage");
    } else {
      // missing file, or an uncalibrated placeholder awaiting its first run
      LS.delCal(S.slug);
      deactivateCalibration();
    }
    // the storage file is the rotation's source of truth for this version
    const rot = cal?.extrinsic?.orientation?.rotate_deg_cw ??
      LS.get(`cvcal:orient:${ss}`,
             LS.get(`cvcal:orient:${S.slug}`, {})).rotate;
    if (rot != null && rot !== S.orient.rotate) {
      S.orient.rotate = rot;
      updateOrientationUI();
      applyOrientationCss();
    }
  } catch { /* store unreachable */ }
  renderCameraInfo();
  updateButtons();
}

$("calNotMine").addEventListener("click", () => {
  S.familyExpanded = true;
  loadCalFamily().then((fam) => renderCalFamily(fam, cleanLabel($("camName").value)));
});
$("calVersionSel").addEventListener("change", (e) => selectCalVersion(e.target.value));
$("calAddLabel").addEventListener("click", async () => {
  const label = cleanLabel(prompt(
    "Label for this camera (1–3 characters). Write it on the camera body:"));
  if (!label) return;
  const fam = await loadCalFamily();
  if (fam.some((f) => f.label === label)) {
    toast(`Label ${label} already exists — select it from the dropdown instead.`, true);
    return;
  }
  $("camName").value = label;
  LS.delCal(S.slug);
  deactivateCalibration();
  S.familyExpanded = true;
  renderCalFamily(fam.concat([{ label, slug: storageSlug() }]), label);
  renderCameraInfo();
  toast(`New camera “${label}” — collect and calibrate to create its file.`);
});
$("calRename").addEventListener("click", async () => {
  const cur = cleanLabel($("camName").value);
  const curSlug = storageSlug();
  const label = cleanLabel(prompt(
    `New label for the ${cur ? `“${cur}”` : "default"} calibration:`));
  if (!label || label === cur) return;
  const r = await fetch(`api/host/calibrations/${curSlug}/rename`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_slug: `${S.slug}__L${label}`, label }),
  });
  const res = await r.json();
  if (!r.ok) { toast(res.error || "rename failed", true); return; }
  $("camName").value = label;
  S.familyExpanded = true;
  const fam = await loadCalFamily();
  renderCalFamily(fam, label);
  await selectCalVersion(label);
  toast(`Renamed to “${label}”.`);
});

async function renderStorage() {
  if (!HOST) return;
  $("hostStorage").classList.remove("hidden");
  let st;
  try { st = await (await fetch("api/host/storage")).json(); }
  catch { return; }
  S.storageStatus = st;
  const where = st.type === "s3"
    ? `S3 — s3://${esc(st.bucket || "?")}/${esc(st.prefix || "?")}/ — key ${esc(st.access_key_id || "?")}`
    : `directory — ${esc(st.path || "?")}`;
  $("storageStatus").innerHTML =
    (st.ok ? '<span class="badge ok">connected</span> '
           : `<span class="badge warn">error</span> ${esc(st.error || "")} — `) + where;
  if (st.config) {
    $("stType").value = st.config.type || "dir";
    $("stDir").value = st.config.dir_path || "";
    $("stEnv").value = st.config.env_file || "";
  }
  toggleStorageFields();
}

function toggleStorageFields() {
  const s3 = $("stType").value === "s3";
  $("stDirLabel").classList.toggle("hidden", s3);
  $("stEnvLabel").classList.toggle("hidden", !s3);
  $("stReveal").disabled = !s3;
}
$("stType").addEventListener("change", toggleStorageFields);

$("stApply").addEventListener("click", async () => {
  const body = { type: $("stType").value, dir_path: $("stDir").value.trim(),
                 env_file: $("stEnv").value.trim() };
  const r = await fetch("api/host/storage", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  const st = await r.json();
  toast(st.ok ? "Storage configured." : "Storage error: " + (st.error || ""), !st.ok);
  renderStorage();
});

$("stReveal").addEventListener("click", async () => {
  const r = await (await fetch("api/host/storage/reveal")).json();
  const el = $("stRevealOut");
  if (el.classList.toggle("hidden")) return;
  el.textContent = r.error ? r.error :
`AWS_ACCESS_KEY_ID=${r.access_key_id}
AWS_SECRET_ACCESS_KEY=${r.secret_access_key}
AWS_DEFAULT_REGION=${r.region}
# identity: ${r.identity}
# files:    s3://${r.bucket}/${r.prefix}/*.json`;
});

async function storePutCalibration(cal) {
  const ss = storageSlug();
  if (!ss) return { error: "no camera selected" };
  try {
    const r = await fetch(`api/host/calibrations/${ss}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cal) });
    return await r.json();
  } catch (e) { return { error: e.message }; }
}

async function detectHost() {
  try {
    const r = await fetch("api/host/ping", { signal: AbortSignal.timeout(2000) });
    if (r.ok && (await r.json()).mode === "host") HOST = true;
  } catch { /* static hosting (e.g. GitHub Pages) — browser mode */ }
}

async function refreshDevices() {
  let entries, havePermission = true;
  if (HOST) {
    $("grantBtn").style.display = "none";
    S.hostCams = await (await fetch("api/host/cameras")).json();
    entries = S.hostCams.map((c) => ({
      value: "host:" + c.node,
      label: `${c.name} (/dev/video${c.node})` +
             (c.duplicate ? " ⚠ duplicate ID — use labels"
                          : c.serial_trusted ? "" : " ⚠ label required"),
      slug: c.slug,
    }));
  } else {
    const devs = await navigator.mediaDevices.enumerateDevices();
    S.devices = devs.filter((d) => d.kind === "videoinput");
    havePermission = S.devices.some((d) => d.label);
    $("grantBtn").style.display = havePermission ? "none" : "";
    entries = S.devices.map((d, i) => ({
      value: d.deviceId,
      label: d.label || `camera ${i + 1}`,
      slug: slugify(d.label),
    }));
  }
  for (const sel of [$("cameraSelect"), $("mCameraSelect")]) {
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select a camera —</option>';
    for (const e of entries) {
      const opt = document.createElement("option");
      opt.value = e.value;
      opt.textContent = e.label + (LS.cal(e.slug) ? "  ✔ calibrated" : "");
      sel.appendChild(opt);
    }
    if (prev) sel.value = prev;
  }
  return havePermission;
}

$("grantBtn").addEventListener("click", async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    s.getTracks().forEach((t) => t.stop());
    await refreshDevices();
    toast("Camera access granted — pick a camera.");
  } catch (e) {
    toast("Camera permission denied: " + e.message, true);
  }
});

$("cameraSelect").addEventListener("change", (e) => {
  if (e.target.value) selectCamera(e.target.value);
});
$("mCameraSelect").addEventListener("change", (e) => {
  if (e.target.value) selectCamera(e.target.value);
});

const RES_PRESETS = [
  [640, 480], [800, 600], [1024, 768], [1280, 720], [1600, 896],
  [1920, 1080], [2560, 1440], [3840, 2160],
];

async function selectCamera(deviceId, width, height) {
  stopCollecting(true);
  let label;
  if (HOST) {
    const node = parseInt(String(deviceId).replace("host:", ""), 10);
    const cam = (S.hostCams || []).find((c) => c.node === node);
    if (!cam) return;
    S.hostCam = await (await fetch(`api/host/cameras/${node}/details`)).json();
    S.device = null;
    S.slug = S.hostCam.slug;
    label = S.hostCam.name;
  } else {
    const dev = S.devices.find((d) => d.deviceId === deviceId);
    if (!dev) return;
    S.device = dev;
    S.hostCam = null;
    S.slug = slugify(dev.label);
    label = dev.label;
  }
  // restore this camera's saved orientation (calibration extrinsic wins)
  const calO = LS.cal(S.slug)?.extrinsic?.orientation;
  S.orient = { rotate: (calO ? calO.rotate_deg_cw
                             : LS.get(`cvcal:orient:${S.slug}`, {}).rotate) || 0 };
  updateOrientationUI();
  await openStream(width || 1280, height || 720);
  if (!HOST && !S.stream) return;
  setTimeout(applyOrientationCss, 150);   // after layout settles
  populateModes();
  renderCameraInfo();
  $("camName").value = LS.get(`cvcal:name:${S.slug}`, "") || "";
  loadImages();
  $("cameraSelect").value = deviceId;
  $("mCameraSelect").value = deviceId;
  // camera's stored calibration becomes active unless a file was uploaded
  if (S.activeCalSource !== "uploaded file") {
    const cal = LS.cal(S.slug);
    if (cal) activateCalibration(cal, "storage");
    else deactivateCalibration();
  }
  applySavedMeasureBoard();
  if (HOST) {
    renderStorage();
    S.familyExpanded = false;
    const fam = await loadCalFamily();
    let pick = fam[0] || null;
    const rem = rigGet(portKey(S.hostCam), undefined);
    if (rem !== undefined) {
      const f = fam.find((x) => (x.label || "") === rem);
      if (f) pick = f;
    }
    renderCalFamily(fam, pick ? pick.label : null);
    if (pick && S.activeCalSource !== "uploaded file") {
      await selectCalVersion(pick.label);
    }
  }
  updateButtons();
}

/* Measurement-board params persist per camera: default to the calibration
   board, but the last board used with this camera wins. */
function applySavedMeasureBoard() {
  const sv = S.slug ? LS.get(`cvcal:mboard:${S.slug}`, null) : null;
  if (!sv || !sv.sx) return false;       // ignore pre-charuco saved boards
  $("mCols").value = sv.sx;
  $("mRows").value = sv.sy;
  $("mSquareSize").value = sv.square;
  $("mMarkerSize").value = sv.marker;
  if (sv.units) $("mUnits").value = sv.units;
  return true;
}

async function openStream(width, height) {
  if (HOST) return openHostStream(width, height);
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
  // exact first so a mode the camera can't deliver fails loudly instead of
  // silently downgrading; fall back to ideal and report what we really got
  const dev = { deviceId: { exact: S.device.deviceId } };
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { ...dev, width: { exact: width }, height: { exact: height } },
    });
  } catch {
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({
        video: { ...dev, width: { ideal: width }, height: { ideal: height } },
      });
    } catch (e) {
      toast("Could not open camera: " + e.message, true);
      renderCameraInfo();
      return;
    }
  }
  const track = S.stream.getVideoTracks()[0];
  S.trackSettings = track.getSettings();
  S.trackCaps = track.getCapabilities ? track.getCapabilities() : null;
  const st = S.trackSettings;
  if (st.width !== width || st.height !== height) {
    toast(`Camera delivered ${st.width}×${st.height} ` +
          `(requested ${width}×${height}).`, true);
  }
  const audio = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.kind === "audioinput");
  S.micPresent = !!st.groupId && audio.some((a) => a.groupId === st.groupId);
  $("liveVideo").srcObject = S.stream;
  $("measVideo").srcObject = S.stream;
  $("liveOverlayMsg").classList.add("hidden");
  $("mStreamInfo").textContent =
    `streaming ${st.width}×${st.height} @ ${Math.round(st.frameRate || 0)} fps`;
  syncModeSelects();
}

/* Host-mode frames reach the page via fetch-parsed streams painted as
   static blob images — NEVER as a multipart <img src="...mjpg">: Chrome's
   compositor shows those live, but canvas.drawImage() of a multipart image
   returns the connection's FIRST frame forever (with GPU rendering), which
   silently corrupted collection with N copies of one frame. */
async function readFrameStream(url, ctrl, onFrame) {
  const r = await fetch(url, { signal: ctrl.signal });
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const nb = new Uint8Array(buf.length + value.length);
    nb.set(buf);
    nb.set(value, buf.length);
    buf = nb;
    for (;;) {
      const nl = buf.indexOf(10);
      if (nl < 0) break;
      const [node, len] = dec.decode(buf.subarray(0, nl)).split(",").map(Number);
      if (!Number.isFinite(len) || buf.length < nl + 1 + len) break;
      onFrame(node, buf.slice(nl + 1, nl + 1 + len));
      buf = buf.slice(nl + 1 + len);
    }
  }
}

const LIVE = { abort: null, url: null };

function stopLiveReader() {
  if (LIVE.abort) { LIVE.abort.abort(); LIVE.abort = null; }
}

function paintLiveFrame(jpg) {
  const url = URL.createObjectURL(new Blob([jpg], { type: "image/jpeg" }));
  const old = LIVE.url;
  LIVE.url = url;
  const live = $("liveImg"), meas = $("measImg");
  let pending = 2;
  const done = () => { if (--pending === 0 && old) URL.revokeObjectURL(old); };
  live.onload = done;
  meas.onload = done;
  live.src = url;
  meas.src = url;
}

async function startLiveReader(node) {
  stopLiveReader();
  const ctrl = new AbortController();
  LIVE.abort = ctrl;
  while (LIVE.abort === ctrl) {
    try {
      await readFrameStream(
        `api/host/multistream?nodes=${node}&width=8192&quality=88&t=${Date.now()}`,
        ctrl, (n, jpg) => paintLiveFrame(jpg));
    } catch { /* aborted, or bridge went away */ }
    if (LIVE.abort !== ctrl) break;
    await new Promise((res) => setTimeout(res, 1000));   // ended: reconnect
  }
}

function stopHostStream(node) {
  if (node == null) return;
  fetch("api/host/stream/stop", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node }) }).catch(() => {});
}

async function openHostStream(width, height) {
  // switching cameras: release the previous device (grid restarts it if needed)
  if (S.hostStreamNode != null && S.hostStreamNode !== S.hostCam.node) {
    stopHostStream(S.hostStreamNode);
  }
  const r = await fetch("api/host/stream/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node: S.hostCam.node, width, height }),
  });
  const info = await r.json();
  if (!r.ok) {
    toast("Could not open camera: " + (info.error || r.statusText), true);
    renderCameraInfo();
    return;
  }
  S.hostStreamNode = S.hostCam.node;
  S.trackSettings = { width: info.width, height: info.height,
                      frameRate: info.fps };
  S.trackCaps = null;
  S.micPresent = !!S.hostCam.has_microphone;
  if (info.width !== width || info.height !== height) {
    toast(`Camera delivered ${info.width}×${info.height} ` +
          `(requested ${width}×${height}).`, true);
  }
  for (const [imgId, vidId] of [["liveImg", "liveVideo"], ["measImg", "measVideo"]]) {
    $(vidId).classList.add("hidden");
    $(imgId).classList.remove("hidden");
  }
  startLiveReader(S.hostCam.node);
  $("liveOverlayMsg").classList.add("hidden");
  $("mStreamInfo").textContent =
    `streaming ${info.width}×${info.height} @ ${Math.round(info.fps || 0)} fps (host)`;
  syncModeSelects();
}

function streamActive() {
  return HOST ? sourceReady($("liveImg")) || sourceReady($("measImg"))
              : !!S.stream;
}


/* Frozen-stream watchdog (host mode): if grabbed frames stop changing,
   first reconnect the MJPEG <img>, then restart the capture server-side.
   Catches cameras dropping off the bus, which otherwise freezes the view
   on the last frame with no error event. */
const WD = { sig: null, changedAt: 0, stage: 0 };
setInterval(async () => {
  if (!HOST || S.hostStreamNode == null || document.hidden) return;
  const src = liveSource();
  const now = Date.now();
  let sig = null;
  if (sourceReady(src)) {
    const im = grabFrame(src);
    if (im) sig = frameSig(im.data);
  }
  if (sig !== null && sig !== WD.sig) {   // healthy: frames are changing
    WD.sig = sig;
    WD.changedAt = now;
    WD.stage = 0;
    return;
  }
  // unchanged frame OR no frame at all while a stream should be live
  if (!WD.changedAt) { WD.changedAt = now; return; }
  if (WD.stage === 0 && now - WD.changedAt > 5000) {
    WD.stage = 1;
    toast("Live stream stalled — reconnecting…", true);
    startLiveReader(S.hostStreamNode);
  } else if (WD.stage === 1 && now - WD.changedAt > 11000) {
    WD.stage = 2;
    try {
      const r = await fetch("api/host/stream/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node: S.hostStreamNode,
                               width: S.trackSettings.width || 1280,
                               height: S.trackSettings.height || 720 }) });
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      startLiveReader(S.hostStreamNode);
      toast("Camera stream restarted.");
    } catch (e) {
      toast("Could not restart the camera (" + e.message +
            ") — was it unplugged? Re-select it once reconnected.", true);
      refreshDevices();
    }
  }
}, 1500);

/* Active frame sources for grabbing/display (video in browser mode,
   MJPEG <img> in host mode). */
const liveSource = () => (HOST ? $("liveImg") : $("liveVideo"));
const measSource = () => (HOST ? $("measImg") : $("measVideo"));
const sourceReady = (el) =>
  (el.videoWidth || el.naturalWidth || 0) > 0;

function syncModeSelects() {
  const cur = `${S.trackSettings.width}x${S.trackSettings.height}`;
  for (const sel of [$("modeSelect"), $("mModeSelect")]) {
    if (!sel.options.length) continue;
    if (![...sel.options].some((o) => o.value === cur)) {
      const opt = document.createElement("option");
      opt.value = cur;
      opt.textContent = cur.replace("x", " × ") + " (current)";
      sel.appendChild(opt);
    }
    sel.value = cur;
  }
}

function populateModes() {
  let list;
  if (HOST && S.hostCam?.modes?.length) {
    // real enumerated modes from V4L2 (unique resolutions, largest first)
    const seen = new Set();
    list = S.hostCam.modes
      .filter((m) => !seen.has(`${m.width}x${m.height}`) &&
                     seen.add(`${m.width}x${m.height}`))
      .map((m) => [m.width, m.height])
      .sort((a, b) => b[0] * b[1] - a[0] * a[1]);
  } else {
    const caps = S.trackCaps;
    const maxW = caps?.width?.max || 1920;
    const maxH = caps?.height?.max || 1080;
    list = RES_PRESETS.filter(([w, h]) => w <= maxW && h <= maxH);
    if (!list.some(([w, h]) => w === maxW && h === maxH)) list.push([maxW, maxH]);
  }
  for (const sel of [$("modeSelect"), $("mModeSelect")]) {
    sel.innerHTML = "";
    for (const [w, h] of list) {
      const opt = document.createElement("option");
      opt.value = `${w}x${h}`;
      opt.textContent = `${w} × ${h}`;
      sel.appendChild(opt);
    }
    sel.disabled = false;
  }
  syncModeSelects();
}

async function onModeChange(value) {
  const [w, h] = value.split("x").map(Number);
  stopCollecting(true);
  await openStream(w, h);
  renderCameraInfo();
}
$("modeSelect").addEventListener("change", (e) => onModeChange(e.target.value));
$("mModeSelect").addEventListener("change", (e) => onModeChange(e.target.value));

function renderCameraInfo() {
  if (HOST && S.hostCam) return renderHostCameraInfo();
  if (!S.device) return;
  const d = S.device, st = S.trackSettings, caps = S.trackCaps;
  const vidpid = (d.label.match(/\(([0-9a-f]{4}:[0-9a-f]{4})\)/i) || [])[1];
  const cal = LS.cal(S.slug);
  const rows = [
    ["Device label", d.label || "—"],
    ["USB VID:PID", vidpid || "not exposed by browser"],
    ["Device ID", (d.deviceId || "").slice(0, 16) + "…"],
    ["Group ID", (d.groupId || "").slice(0, 16) + "…"],
    ["Slug (cal filename)", S.slug],
    ["Streaming", st.width ? `${st.width} × ${st.height} @ ${Math.round(st.frameRate || 0)} fps` : "—"],
    ["Max capability", caps?.width ? `${caps.width.max} × ${caps.height.max} @ ${Math.round(caps.frameRate?.max || 0)} fps` : "not reported"],
    ["Microphone", S.micPresent
      ? '<span class="badge ok">present (same device)</span>'
      : '<span class="badge no">none detected</span>', true],
    ["Calibration", cal
      ? `<span class="badge ok">stored in browser</span>`
      : '<span class="badge warn">not calibrated</span>', true],
  ];
  if (cal) {
    const i = cal.intrinsic;
    rows.push(["Calibrated", `${esc((i.calibrated_at || "").slice(0, 19).replace("T", " "))} — RMS ${i.rms_reprojection_error_px?.toFixed(3)} px, ${i.num_images} images @ ${i.image_size?.join("×")}`]);
  }
  rows.push(["Note", '<span class="dim">Browsers can\'t read USB serial numbers — use a label to tell identical cameras apart.</span>', true]);
  $("cameraInfo").innerHTML = "<table>" + rows.map(([k, v, raw]) =>
    `<tr><td>${k}</td><td>${raw ? v : esc(v)}</td></tr>`).join("") + "</table>";
}

function renderHostCameraInfo() {
  const c = S.hostCam, u = c.usb || {}, st = S.trackSettings;
  const cal = LS.cal(S.slug);
  const rows = [
    ["Device name", c.name],
    ["Manufacturer", u.manufacturer || "—"],
    ["Product", u.product || "—"],
    ["USB VID:PID", u.id_vendor ? `${u.id_vendor}:${u.id_product}` : "—"],
    ["Serial", c.serial_trusted
      ? esc(u.serial)
      : `${esc(u.serial || "none")} <span class="badge warn">${c.duplicate
          ? "shared by several connected cameras — use labels"
          : "generic — label required"}</span>`, true],
    ["USB", u.usb_version ? `${u.usb_version.trim()} @ ${u.speed_mbps} Mbps (bus ${u.bus_path})` : "—"],
    ["Device path", c.path],
    ["Stable ID (by-id)", c.by_id ? c.by_id.split("/").pop() : "—"],
    ["Slug (cal filename)", S.slug],
    ["Driver", c.driver?.driver ? `${c.driver.driver} (${c.driver.bus_info})` : "—"],
    ["Streaming", st.width ? `${st.width} × ${st.height} @ ${Math.round(st.frameRate || 0)} fps (host bridge)` : "—"],
    ["Microphone", c.has_microphone
      ? '<span class="badge ok">present</span>'
      : '<span class="badge no">none</span>', true],
    ["Calibration", cal
      ? '<span class="badge ok">saved</span> <span class="dim">(calibration storage + browser copy)</span>'
      : '<span class="badge warn">not calibrated</span>', true],
  ];
  if (cal) {
    const i = cal.intrinsic;
    rows.push(["Calibrated", `${esc((i.calibrated_at || "").slice(0, 19).replace("T", " "))} — RMS ${i.rms_reprojection_error_px?.toFixed(3)} px, ${i.num_images} images @ ${i.image_size?.join("×")}`]);
  }
  let html = "<table>" + rows.map(([k, v, raw]) =>
    `<tr><td>${k}</td><td>${raw ? v : esc(v)}</td></tr>`).join("") + "</table>";
  const modes = c.modes || [];
  if (modes.length) {
    html += `<details class="modes"><summary>${modes.length} video modes</summary><table>` +
      modes.map((m) =>
        `<tr><td>${esc(m.format)}</td><td>${m.width}×${m.height}</td><td class="dim">${m.fps.join(", ")} fps</td></tr>`).join("") +
      "</table></details>";
  }
  $("cameraInfo").innerHTML = html;
}

/* ------------------------------------------------------------ frame grab */
/* Stream orientation: rotate (cw), then flips in displayed axes. Applied to
   every grabbed frame, so collection, calibration, undistortion, snaps and
   measurement all operate on oriented frames. */
S.orient = { rotate: 0 };
/* View-only mirror for the collect live view (holding a board in front of
   the camera). Never persisted, never applied to grabbed frames — rotation
   is a real extrinsic, mirroring is not. */
S.viewFlip = { h: false, v: false };

function orientationExtrinsic() {
  return {
    rotate_deg_cw: S.orient.rotate,
    note: "rotate raw frames clockwise by this before undistortion; " +
          "adjust downstream if the operational mounting differs",
  };
}

function applyOrientationCss() {
  const { rotate } = S.orient;
  for (const id of ["liveVideo", "measVideo", "liveImg", "measImg"]) {
    const el = $(id);
    let k = 1;
    if (rotate % 180 !== 0 && el.clientWidth) k = el.clientHeight / el.clientWidth;
    // leftmost transforms act in screen space: the mirror applies to the
    // final displayed image whatever the rotation is (collect view only)
    const flip = (id === "liveVideo" || id === "liveImg")
      ? (S.viewFlip.h ? "scaleX(-1) " : "") + (S.viewFlip.v ? "scaleY(-1) " : "")
      : "";
    el.style.transform = `${flip}scale(${k}) rotate(${rotate}deg)`;
  }
}

for (const [btn, axis] of [["flipH", "h"], ["flipV", "v"]]) {
  $(btn).addEventListener("click", () => {
    S.viewFlip[axis] = !S.viewFlip[axis];
    $(btn).classList.toggle("active-mode", S.viewFlip[axis]);
    applyOrientationCss();
  });
}

function updateOrientationUI() {
  $("oRotate").value = String(S.orient.rotate);
}

function setOrientation(change) {
  S.orient = { ...S.orient, ...change };
  updateOrientationUI();
  applyOrientationCss();
  if (!S.slug) return;
  LS.set(`cvcal:orient:${storageSlug()}`, S.orient);
  const cal = LS.cal(S.slug);
  if (cal) {
    cal.extrinsic = { ...(cal.extrinsic || {}), orientation: orientationExtrinsic() };
    LS.setCal(S.slug, cal);
    if (S.activeCal?.slug === S.slug) S.activeCal.extrinsic = cal.extrinsic;
    if (S.lastResult?.slug === S.slug) S.lastResult.extrinsic = cal.extrinsic;
    if (HOST) storePutCalibration(cal);
    toast("Rotation saved with this camera's calibration." +
      (S.orient.rotate % 180 !== 0
        ? " Recalibrate at this rotation for best accuracy."
        : ""));
  }
}

$("oRotate").addEventListener("change", (e) => {
  setOrientation({ rotate: +e.target.value });
  if (S.hostCam) rigSet(portRotKey(S.hostCam), +e.target.value);
});

const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });
function grabFrame(source) {
  const vw = source.videoWidth || source.naturalWidth;
  const vh = source.videoHeight || source.naturalHeight;
  if (!vw || !vh) return null;
  const { rotate } = S.orient;
  const swap = rotate % 180 !== 0;
  const ow = swap ? vh : vw, oh = swap ? vw : vh;
  grabCanvas.width = ow; grabCanvas.height = oh;
  grabCtx.save();
  grabCtx.translate(ow / 2, oh / 2);
  grabCtx.rotate(rotate * Math.PI / 180);
  grabCtx.drawImage(source, -vw / 2, -vh / 2);
  grabCtx.restore();
  return grabCtx.getImageData(0, 0, ow, oh);
}

function makeThumb(imageData, targetW = 240) {
  const scale = targetW / imageData.width;
  const c = document.createElement("canvas");
  c.width = targetW;
  c.height = Math.round(imageData.height * scale);
  const src = document.createElement("canvas");
  src.width = imageData.width; src.height = imageData.height;
  src.getContext("2d").putImageData(imageData, 0, 0);
  c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.72);
}

/* -------------------------------------------------------------- collection */
function updateButtons() {
  const ready = streamActive() && S.pyReady;
  $("collectBtn").disabled = !ready;
  $("clearBtn").disabled = !S.slug || (S.images.length === 0 && !S.collecting);
  $("calibrateBtn").disabled = !ready || S.images.length < 5 || S.collecting;
  $("snapRawBtn").disabled = !ready;
  $("snapUndBtn").disabled = !ready || !S.activeCal;
  $("downloadCalBtn").disabled = !S.lastResult;
  $("pruneRerunBtn").disabled = !S.lastResult || !S.lastRun;
  $("downloadActiveBtn").disabled = !S.activeCal;
  $("clearCalBtn").disabled = !S.activeCal;
}

function updateCountIndicator() {
  const n = S.images.length;
  const light = $("statusLight");
  light.className = "light" +
    (n >= 40 ? " green" : n >= 30 ? " teal" : n >= 20 ? " yellow" : "");
  let quality = "";
  if (n >= 40) quality = " — ideal ✔";
  else if (n >= 30) quality = " — good";
  else if (n >= 20) quality = " — minimum reached";
  else if (S.collecting) quality = ` — need ${20 - n} more for minimum`;
  $("statusText").textContent = `${n} image${n === 1 ? "" : "s"}${quality}`;
  updateButtons();
}

$("collectBtn").addEventListener("click", () =>
  S.collecting ? stopCollecting() : startCollecting());

function startCollecting() {
  if (!streamActive() || !S.pyReady) return;
  S.collecting = true;
  $("liveWrap").classList.add("collecting");
  $("collectBtn").textContent = "⏹ Stop Collecting (Space)";
  $("collectBtn").classList.add("stop");
  switchTab("collect");
  let remain = 5;
  const cd = $("countdown");
  cd.classList.remove("hidden");
  cd.textContent = remain;
  const tick = setInterval(() => {
    remain -= 1;
    if (remain <= 0) {
      clearInterval(tick);
      cd.classList.add("hidden");
      snapCalibImage();
      S.collectTimers.push(setInterval(snapCalibImage, 2000));
    } else {
      cd.textContent = remain;
    }
  }, 1000);
  S.collectTimers.push(tick);
  updateCountIndicator();
}

function stopCollecting(silent = false) {
  if (!S.collecting && !S.collectTimers.length) return;
  S.collecting = false;
  S.collectTimers.forEach(clearInterval);
  S.collectTimers = [];
  $("countdown").classList.add("hidden");
  $("liveWrap").classList.remove("collecting");
  $("collectBtn").textContent = "📸 Collect Calibration Images";
  $("collectBtn").classList.remove("stop");
  updateCountIndicator();
  if (!silent) toast(`Stopped — ${S.images.length} images collected.`);
}

/* Live cameras never produce byte-identical frames (sensor noise), so an
   identical grab means the stream is frozen — e.g. the camera dropped off
   the USB bus mid-collection. Never let duplicates into the image set. */
function frameSig(data) {
  let h = 0;
  for (let i = 0; i < data.length; i += 997) h = (h * 31 + data[i]) | 0;
  return h;
}

let snapInFlight = false;
async function snapCalibImage(manual = false) {
  if (snapInFlight || !S.pyReady || (!manual && !S.collecting)) return;
  snapInFlight = true;
  try {
    const im = grabFrame(liveSource());
    if (!im) return;
    const sig = frameSig(im.data);
    if (S.images.some((r) => r.sig === sig)) {
      const badge = $("shotBadge");
      badge.textContent = "✖ duplicate frame — stream frozen?";
      badge.className = "shot-badge bad";
      setTimeout(() => badge.classList.add("hidden"), 1500);
      S.dupRun = (S.dupRun || 0) + 1;
      if (S.collecting && S.dupRun >= 3) {
        stopCollecting(true);
        toast("The live stream appears frozen (identical frames) — " +
              "collection stopped. Check the camera connection.", true);
      }
      return;
    }
    S.dupRun = 0;
    if (S.images.length &&
        (S.images[0].w !== im.width || S.images[0].h !== im.height)) {
      toast(`Resolution changed (collection is ${S.images[0].w}×${S.images[0].h}, ` +
            `stream is ${im.width}×${im.height}) — clear all or switch back.`, true);
      stopCollecting();
      return;
    }
    const { sx, sy, square, marker } = settings();
    const det = JSON.parse(py.detect_charuco(
      im.data, im.width, im.height, sx, sy, square, marker));
    const fl = $("flash");
    fl.classList.remove("on");
    void fl.offsetWidth;
    fl.classList.add("on");
    const found = !!det.n;
    const badge = $("shotBadge");
    // on a reject, say which kind: no markers decoded at all (board out of
    // frame / wrong board) vs. some decoded but too few corners (board too
    // small, too oblique or motion-blurred to resolve)
    badge.textContent = found
      ? `✔ board detected (${det.n} corners)`
      : det.markers
        ? `✖ discarded — only ${det.markers} marker${det.markers === 1 ? "" : "s"}` +
          ` decoded, need ${det.min_corners} corners — move closer / hold steadier`
        : "✖ discarded — no board markers found";
    badge.className = "shot-badge " + (found ? "good" : "bad");
    setTimeout(() => badge.classList.add("hidden"), 1500);
    if (found) {
      const rec = {
        id: Date.now() + "-" + Math.random().toString(36).slice(2, 6),
        ts: new Date().toISOString(),
        w: im.width, h: im.height,
        sx, sy, square, marker, sig,
        corners: det.corners, ids: det.ids,
        thumb: makeThumb(im),
      };
      S.images.unshift(rec);               // newest first
      if (LS.setImages(storageSlug(), S.images)) {
        addThumb(rec, true);
      } else {
        S.images.shift();                  // storage full — roll back
      }
    }
    updateCountIndicator();
  } finally {
    snapInFlight = false;
  }
}

function loadImages() {
  S.images = LS.images(storageSlug());
  // legacy/order safety: newest first
  S.images.sort((a, b) => (a.ts < b.ts ? 1 : -1));
  $("thumbGrid").innerHTML = "";
  S.images.forEach((rec) => addThumb(rec, false));
  updateCountIndicator();
}

function addThumb(rec, prepend) {
  const div = document.createElement("div");
  div.className = "thumb";
  div.dataset.id = rec.id;
  div.innerHTML = `
    <img src="${rec.thumb}" loading="lazy" alt="">
    <span class="mark ok">✔</span>
    <div class="acts">
      <button title="View" data-act="view">🔍</button>
      <button title="Delete" data-act="del">🗑</button>
    </div>`;
  div.querySelector('[data-act="view"]').addEventListener("click", () =>
    openLightbox(rec.thumb));
  div.querySelector('[data-act="del"]').addEventListener("click", () => {
    S.images = S.images.filter((r) => r.id !== rec.id);
    LS.setImages(storageSlug(), S.images);
    div.remove();
    updateCountIndicator();
  });
  const grid = $("thumbGrid");
  if (prepend && grid.firstChild) grid.insertBefore(div, grid.firstChild);
  else grid.appendChild(div);
}

$("clearBtn").addEventListener("click", () => {
  if (!S.slug) return;
  if (!confirm(`Delete all ${S.images.length} collected images for this camera?`)) return;
  stopCollecting(true);
  S.images = [];
  LS.setImages(storageSlug(), []);
  $("thumbGrid").innerHTML = "";
  updateCountIndicator();
  toast("All collected images deleted.");
});

/* ------------------------------------------------------------ calibration */
const STEPS = [
  ["gather", "Gather collected corner sets"],
  ["solve", "Solve camera matrix + distortion"],
  ["analyze", "Reprojection error analysis"],
  ["save", "Save calibration"],
];
const STEP_ICON = { pending: "○", active: "◐", done: "✔", fail: "✖" };
let stepState = {};

function resetSteps() {
  stepState = Object.fromEntries(STEPS.map(([k]) => [k, ["pending", ""]]));
  renderSteps();
  $("calibLog").textContent = "";
  $("calibResult").classList.add("hidden");
  $("errChart").innerHTML = "";
  $("reprojWrap").innerHTML = "";
}
function setStep(key, status, detail = null) {
  if (detail !== null) stepState[key] = [status, detail];
  else stepState[key][0] = status;
  renderSteps();
}
function renderSteps() {
  $("stepList").innerHTML = STEPS.map(([k, label]) => {
    const [status, detail] = stepState[k];
    return `<li class="${status}">
      <span class="st-icon">${STEP_ICON[status]}</span>
      <span>${label}</span><span class="st-detail">${esc(detail)}</span></li>`;
  }).join("");
}
function log(msg) {
  const el = $("calibLog");
  el.textContent += `[${new Date().toTimeString().slice(0, 8)}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}
const tick = () => new Promise((r) => setTimeout(r, 30));

$("calibrateBtn").addEventListener("click", runCalibration);

async function runCalibration() {
  stopCollecting(true);
  const st = settings();
  if (!st.square || !st.marker || !st.sx || !st.sy) {
    toast("Set the board's squares and square/marker sizes first.", true);
    return;
  }
  switchTab("results");
  resetSteps();
  try {
    // gather -----------------------------------------------------------
    setStep("gather", "active", "");
    await tick();
    const legacy = S.images.filter((r) => !r.ids).length;
    const boardMatch = S.images.filter((r) => r.ids &&
      r.sx === st.sx && r.sy === st.sy &&
      r.square === st.square && r.marker === st.marker);
    if (!boardMatch.length) throw new Error(
      "No stored images match this ChArUco board — collect images first" +
      (legacy ? " (old checkerboard captures can't be reused — clear them)"
              : "."));
    const ref = boardMatch[0];
    const usable = boardMatch.filter((r) => r.w === ref.w && r.h === ref.h);
    const skipped = S.images.length - usable.length;
    const nCorners = usable.reduce((a, r) => a + r.ids.length, 0);
    log(`${S.images.length} stored images; ${usable.length} match ` +
        `${st.sx}×${st.sy} board @ ${ref.w}×${ref.h} ` +
        `(${nCorners} corners total)` +
        (skipped ? ` — ${skipped} skipped (different board/resolution` +
                   (legacy ? " or old checkerboard captures" : "") + ")" : ""));
    if (usable.length < 5) throw new Error(
      `Only ${usable.length} usable views — collect more, or check the ` +
      `board settings match what was used during collection.`);
    // oldest-first for stable indexing in results
    const ordered = [...usable].reverse();
    setStep("gather", "done", `${usable.length} views`);

    // solve --------------------------------------------------------------
    setStep("solve", "active", "this can take a few seconds…");
    log(`Calibrating at ${ordered[0].w}×${ordered[0].h} with ${ordered.length} views…`);
    await tick();
    const res = JSON.parse(py.calibrate_charuco(
      JSON.stringify(ordered.map((r) => ({ corners: r.corners, ids: r.ids }))),
      ordered[0].w, ordered[0].h, st.sx, st.sy, st.square, st.marker));
    if (res.error) throw new Error(res.error);
    if (res.duplicate_views_removed) {
      log(`WARNING: ${res.duplicate_views_removed} duplicate frames ` +
          `discarded — the camera stream may have frozen during collection.`);
    }
    // remember which stored image produced each per-view error (for pruning)
    S.lastRun = { ids: res.used_indices.map((i) => ordered[i].id),
                  perView: res.per_view };
    log(`RMS reprojection error: ${res.rms} px`);
    setStep("solve", "done", `RMS ${res.rms} px`);

    // analyze ------------------------------------------------------------
    setStep("analyze", "active");
    await tick();
    renderErrChart(res, ordered);
    renderReprojArtifacts(res, ordered);
    const lo = Math.min(...res.per_view), hi = Math.max(...res.per_view);
    log(`Per-view error range: ${lo.toFixed(3)} – ${hi.toFixed(3)} px`);
    setStep("analyze", "done", `${lo.toFixed(3)} – ${hi.toFixed(3)} px`);

    // save ---------------------------------------------------------------
    setStep("save", "active");
    await tick();
    const cal = buildCalibrationJson(res, ordered, st);
    S.lastResult = cal;
    LS.setCal(S.slug, cal);
    activateCalibration(cal, "calibrated");
    S.lastStoredAt = null;
    if (HOST) {
      const sres = await storePutCalibration(cal);
      if (sres.ok) S.lastStoredAt = sres.location;
      log(sres.ok ? `Saved to storage: ${sres.location}`
                  : `Storage save skipped/failed: ${sres.error || sres.skipped}`);
    }
    renderResultCard(res, cal);
    renderCameraInfo();
    await refreshDevices();
    log(`Saved to browser storage as "${S.slug}". Use Download to export the JSON.`);
    setStep("save", "done", "stored in browser");
    toast("Calibration complete — download the JSON or go measure!");
  } catch (e) {
    for (const [k] of STEPS) if (stepState[k][0] === "active") setStep(k, "fail");
    log("ERROR: " + e.message);
    toast("Calibration failed: " + e.message, true);
  }
  updateButtons();
}

function buildCalibrationJson(res, ordered, st) {
  const existing = LS.cal(S.slug);
  const capture_settings = {
    width: S.trackSettings.width, height: S.trackSettings.height,
    frame_rate: S.trackSettings.frameRate,
  };
  const camera = HOST && S.hostCam ? {
    platform: "host-bridge",
    label: S.hostCam.name,
    usb: S.hostCam.usb,
    serial_trusted: !!S.hostCam.serial_trusted,
    by_id: S.hostCam.by_id,
    driver: S.hostCam.driver || null,
    modes: S.hostCam.modes || [],
    microphone: S.micPresent,
    capture_settings,
  } : {
    platform: "web",
    label: S.device.label,
    usb_vid_pid: (S.device.label.match(/\(([0-9a-f]{4}:[0-9a-f]{4})\)/i) || [])[1] || null,
    device_id: S.device.deviceId,
    group_id: S.device.groupId,
    microphone: S.micPresent,
    capture_settings,
    user_agent: navigator.userAgent,
  };
  if (HOST && S.hostCam) {
    camera.assigned_label = cleanLabel(st.name) || null;
    camera.serial_generic = !S.hostCam.serial_trusted;
  }
  return {
    schema_version: 1,
    name: st.name || (HOST ? S.hostCam?.name : S.device?.label) || S.slug,
    slug: storageSlug() || S.slug,
    camera,
    intrinsic: {
      calibrated_at: new Date().toISOString(),
      image_size: res.image_size,
      camera_matrix: res.camera_matrix,
      dist_coeffs: res.dist_coeffs,
      distortion_model: "opencv_plumb_bob",
      rms_reprojection_error_px: res.rms,
      // std devs of (fx, fy, cx, cy, k1, k2, p1, p2, k3) from
      // calibrateCameraExtended — large values flag poor coverage
      intrinsic_std_deviations: res.std_intrinsics,
      angular: res.angular,
      per_view_errors_px: res.per_view.map((e, i) => ({
        index: i, ts: ordered[i]?.ts, error_px: e,
        n_corners: res.per_view_corners?.[i] })),
      num_images: res.per_view.length,
      board: {
        type: "charuco",
        dictionary: "DICT_APRILTAG_36h11",
        squares_x: st.sx,
        squares_y: st.sy,
        square_length_mm: st.square,
        marker_length_mm: st.marker,
        square_length_m: +(st.square / 1000).toFixed(6),
        marker_length_m: +(st.marker / 1000).toFixed(6),
        border_bits: 1,
        legacy_pattern: false,
      },
    },
    extrinsic: { ...(existing?.extrinsic || {}),
                 orientation: orientationExtrinsic() },
  };
}

function renderResultCard(res, cal) {
  const K = res.camera_matrix;
  const rmsCls = res.rms < 0.5 ? "good" : res.rms < 1.0 ? "okay" : "poor";
  $("calibResult").classList.remove("hidden");
  $("calibResult").innerHTML = `
    <div>RMS reprojection error</div>
    <div class="rms ${rmsCls}">${res.rms} px</div>
    <table style="margin-top:8px">
      <tr><td class="dim">Focal length&nbsp;</td><td>fx=${K[0][0].toFixed(2)}, fy=${K[1][1].toFixed(2)} px</td></tr>
      <tr><td class="dim">Principal point&nbsp;</td><td>(${K[0][2].toFixed(2)}, ${K[1][2].toFixed(2)})</td></tr>
      <tr><td class="dim">Distortion&nbsp;</td><td><code>[${res.dist_coeffs.map((d) => d.toFixed(4)).join(", ")}]</code></td></tr>
      <tr><td class="dim">Field of view&nbsp;</td><td>${res.angular.fov_degrees.horizontal.toFixed(1)}° × ${res.angular.fov_degrees.vertical.toFixed(1)}° (diag ${res.angular.fov_degrees.diagonal.toFixed(1)}°)</td></tr>
      <tr><td class="dim">Angular res.&nbsp;</td><td>${res.angular.degrees_per_pixel_at_center.x.toFixed(5)}°/px at center (undistorted; falls off cos²θ off-axis)</td></tr>
      <tr><td class="dim">Images used&nbsp;</td><td>${res.per_view.length} @ ${res.image_size.join("×")}</td></tr>
      <tr><td class="dim">Stored as&nbsp;</td><td><code>${esc(cal.slug)}</code> — ${
        S.lastStoredAt
          ? `<code>${esc(S.lastStoredAt)}</code> <span class="dim">+ browser copy</span>`
          : "browser localStorage"}</td></tr>
    </table>`;
}

const ERR_AXIS_MAX = 3.0;   // fixed axis; anything above clamps red at 3.0
function renderErrChart(res, ordered) {
  const guides = `<span class="guide g05" title="0.5 px — ideal"></span>` +
                 `<span class="guide g10" title="1.0 px — removal candidate"></span>`;
  const rows = res.per_view.map((e, i) => {
    const cls = e > 1 ? "poor" : e > 0.5 ? "okay" : "";
    const label = (ordered[i]?.ts || `view ${i}`).slice(5, 19).replace("T", " ");
    const pct = (100 * Math.min(e, ERR_AXIS_MAX) / ERR_AXIS_MAX).toFixed(1);
    return `<div class="err-row">
      <span class="fname">#${i + 1} — ${esc(label)}</span>
      <span class="bar-track">${guides}<span class="bar ${cls}" style="width:${pct}%"></span></span>
      <span>${e.toFixed(3)}${e > ERR_AXIS_MAX ? " ▸" : ""}</span></div>`;
  }).join("");
  $("errChart").innerHTML = `<div class="err-row axis">
      <span></span>
      <span class="axis-track">${guides}<span class="ax al">0</span><span class="ax a05">0.5</span><span class="ax a10">1.0</span><span class="ax ar">3 px</span></span>
      <span></span></div>` + rows;
}

function renderReprojArtifacts(res, ordered) {
  const order = res.per_view.map((e, i) => [e, i]).sort((a, b) => a[0] - b[0]);
  const wrap = $("reprojWrap");
  wrap.innerHTML = "";
  for (const [label, [err, idx]] of [["Best", order[0]], ["Worst", order[order.length - 1]]]) {
    const rec = ordered[idx];
    if (!rec) continue;
    const fig = document.createElement("figure");
    fig.innerHTML = `<figcaption>${label} view — #${idx + 1} (${err.toFixed(3)} px)</figcaption>`;
    const canvas = document.createElement("canvas");
    fig.appendChild(canvas);
    wrap.appendChild(fig);
    const img = new Image();
    img.onload = () => {
      const scaleUp = 2.4;                       // thumbs are small; enlarge
      canvas.width = img.width * scaleUp;
      canvas.height = img.height * scaleUp;
      const ctx = canvas.getContext("2d");
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const s = canvas.width / rec.w;            // full-res -> canvas coords
      for (let i = 0; i < rec.corners.length; i++) {
        const [dx, dy] = rec.corners[i];
        const [rx, ry] = res.reprojected[idx][i];
        ctx.strokeStyle = "#00dd00"; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(dx * s, dy * s, 5, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = "#ff00ff";
        ctx.beginPath();
        ctx.moveTo(rx * s - 5, ry * s); ctx.lineTo(rx * s + 5, ry * s);
        ctx.moveTo(rx * s, ry * s - 5); ctx.lineTo(rx * s, ry * s + 5);
        ctx.stroke();
      }
    };
    img.src = rec.thumb;
  }
}

$("pruneRerunBtn").addEventListener("click", () => {
  if (!S.lastRun) return;
  const thr = parseFloat($("errThreshold").value) || 1.0;
  const badIds = new Set(
    S.lastRun.ids.filter((id, i) => S.lastRun.perView[i] > thr));
  if (!badIds.size) {
    toast(`No images above ${thr} px — nothing to remove.`);
    return;
  }
  if (S.images.filter((r) => !badIds.has(r.id)).length < 5) {
    toast(`Removing ${badIds.size} image(s) would leave fewer than 5 — ` +
          `lower the threshold or collect more images.`, true);
    return;
  }
  S.images = S.images.filter((r) => !badIds.has(r.id));
  LS.setImages(storageSlug(), S.images);
  loadImages();
  toast(`Removed ${badIds.size} image(s) with error > ${thr} px — recalibrating…`);
  runCalibration();
});

/* ----------------------------------------------------- calibration source */
function pushActiveCalToPython() {
  const i = S.activeCal.intrinsic;
  py.set_active_calibration(
    JSON.stringify(i.camera_matrix), JSON.stringify(i.dist_coeffs),
    i.image_size[0], i.image_size[1]);
}

function activateCalibration(cal, source) {
  if (!cal?.intrinsic?.camera_matrix) {
    toast("That file has no intrinsic.camera_matrix — not a calibration file?", true);
    return;
  }
  S.activeCal = cal;
  S.activeCalSource = source;
  if (S.pyReady) pushActiveCalToPython();
  const i = cal.intrinsic;
  const sourceLabel = { "storage": "saved", "storage backend": "saved",
    "calibrated": "just calibrated", "uploaded file": "loaded file" }[source] || source;
  $("activeCalInfo").innerHTML =
    `<span class="badge ok">active</span> ${esc(cal.name || cal.slug)} ` +
    `<span class="dim">(${sourceLabel}) — RMS ${i.rms_reprojection_error_px?.toFixed(3)} px ` +
    `@ ${i.image_size?.join("×")}</span>`;
  const cb = i.board;
  if (cb?.type === "charuco") {          // legacy checkerboard cals: skip
    $("mCols").value = cb.squares_x;
    $("mRows").value = cb.squares_y;
    $("mSquareSize").value = cb.square_length_mm;
    $("mMarkerSize").value = cb.marker_length_mm;
  }
  applySavedMeasureBoard();   // last board used with this camera wins
  $("undMsg").classList.add("hidden");
  updateButtons();
}

function deactivateCalibration() {
  S.activeCal = null;
  S.activeCalSource = "";
  if (S.pyReady) py.clear_active_calibration();
  $("activeCalInfo").textContent =
    "No calibration active — calibrate a camera or load a calibration file.";
  $("undMsg").classList.remove("hidden");
  updateButtons();
}

$("uploadCalBtn").addEventListener("click", () => $("uploadCalInput").click());
$("uploadCalInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  try {
    const cal = JSON.parse(await file.text());
    activateCalibration(cal, "uploaded file");
    if (cal.slug) LS.setCal(cal.slug, cal);   // keep it for next time
    toast(`Loaded calibration "${cal.name || cal.slug || file.name}".`);
  } catch (err) {
    toast("Could not read calibration file: " + err.message, true);
  }
});

function downloadJson(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

$("downloadCalBtn").addEventListener("click", () =>
  S.lastResult && downloadJson(S.lastResult, `${S.lastResult.slug}.json`));
$("downloadActiveBtn").addEventListener("click", () =>
  S.activeCal && downloadJson(S.activeCal, `${S.activeCal.slug || "calibration"}.json`));
$("clearCalBtn").addEventListener("click", deactivateCalibration);

/* --------------------------------------------------- measure: live views */
const undCanvas = $("undCanvas");
const undCtx = undCanvas.getContext("2d");
let undBusy = false;
setInterval(async () => {
  if (activeTab !== "measure" || !S.activeCal || !S.pyReady || !streamActive() || undBusy) return;
  undBusy = true;
  try {
    const im = grabFrame(measSource());
    if (im) {
      const proxy = py.undistort_frame(im.data, im.width, im.height);
      const u8 = proxy.toJs();
      proxy.destroy?.();
      undCanvas.width = im.width; undCanvas.height = im.height;
      undCtx.putImageData(
        new ImageData(new Uint8ClampedArray(u8.buffer || u8), im.width, im.height), 0, 0);
    }
  } catch (e) { /* transient frame errors are fine */ }
  undBusy = false;
}, 70);

/* ------------------------------------------------------- measure: snaps */
$("snapRawBtn").addEventListener("click", () => takeSnap("raw"));
$("snapUndBtn").addEventListener("click", () => takeSnap("undistorted"));

async function takeSnap(source) {
  if (!S.pyReady || !streamActive()) return;
  let im;
  if (source === "undistorted") {
    if (!undCanvas.width) { toast("Undistorted view not ready yet.", true); return; }
    im = undCtx.getImageData(0, 0, undCanvas.width, undCanvas.height);
  } else {
    im = grabFrame(measSource());
  }
  if (!im) { toast("No frame available.", true); return; }
  S.snap = { imageData: im, w: im.width, h: im.height, source };
  S.measurements = [];
  S.measurePts = [];
  setMeasuring(false);
  clearRect();                       // stale top-down view belongs to the old snap
  const c = $("snapImg");
  c.width = im.width; c.height = im.height;
  c.getContext("2d").putImageData(im, 0, 0);
  const oc = $("snapCanvas");
  oc.width = im.width; oc.height = im.height;
  $("snapPanel").style.display = "";
  $("measureList").innerHTML = "";
  $("snapLabel").textContent = `${source} frame — ${im.width}×${im.height} — detecting board…`;
  $("snapMeasureBtn").disabled = true;
  await tick();
  prepareBoard();
  $("snapPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function prepareBoard() {
  if (!S.snap) return;
  const m = measureSettings();
  const res = JSON.parse(py.prepare_measure(
    S.snap.imageData.data, S.snap.w, S.snap.h,
    m.sx, m.sy, m.square, m.marker,
    S.snap.source === "undistorted"));
  S.snapBoard = res.found ? res.corners : null;
  $("snapLabel").textContent =
    `${S.snap.source} frame — ${S.snap.w}×${S.snap.h} — ` +
    (res.found ? `ChArUco board detected (${res.n} corners)` : "no board");
  $("snapMeasureBtn").disabled = !res.found;
  $("genOrthoBtn").disabled = !res.found;
  if (!res.found && S.measuring) setMeasuring(false);
  renderPose(res.found ? res.pose : null, res.found);
  drawMeasureOverlay();
}

/* Camera<->board pose (solvePnP) — shown whenever a calibrated snap sees
   the board. This is the same math the later extrinsic stage will use. */
function renderPose(pose, boardFound) {
  const el = $("posePanel");
  if (!boardFound) { el.style.display = "none"; return; }
  el.style.display = "";
  if (!pose) {
    el.innerHTML = '<span class="dim">Board found, but computing its pose ' +
      "(distance / orientation) needs an active calibration.</span>";
    return;
  }
  const d = pose.distance_mm;
  const dist = d >= 1000 ? `${(d / 1000).toFixed(3)} m` : `${d.toFixed(0)} mm`;
  const [x, y, z] = pose.position_mm;
  el.innerHTML =
    `<b>Board pose</b> <span class="dim small">(solvePnP on ${pose.n_corners}` +
    ` corners — fit RMS ${pose.reproj_rms_px} px)</span>
    <div class="pose-grid">
      <span>Distance</span><b>${dist}</b>
      <span>Position <span class="dim small">(camera frame, mm: +x right,
        +y down, +z forward)</span></span>
      <b>[${x.toFixed(0)}, ${y.toFixed(0)}, ${z.toFixed(0)}]</b>
      <span>Board tilt <span class="dim small">(0° = facing the camera)</span></span>
      <b>${pose.tilt_deg.toFixed(1)}°</b>
      <span>Off-axis <span class="dim small">(board center vs optical axis)</span></span>
      <b>${pose.off_axis_deg.toFixed(1)}°</b>
      <span>Orientation <span class="dim small">(yaw / pitch / roll)</span></span>
      <b>${pose.yaw_deg.toFixed(1)}° / ${pose.pitch_deg.toFixed(1)}° / ${pose.roll_deg.toFixed(1)}°</b>
    </div>`;
}

/* Board-param changes while a snap is open: re-fit the homography and
   recompute every stored measurement from its original click points, so
   existing annotations update in place (e.g. after a unit switch). */
function remeasureAll() {
  const pairs = S.measurements.map((m) => [m.p1, m.p2]);
  S.measurements = [];
  if (S.snapBoard) {
    for (const [p1, p2] of pairs) {
      const res = JSON.parse(py.measure_points(p1[0], p1[1], p2[0], p2[1]));
      if (res.error) continue;
      S.measurements.push({ p1, p2, distance_mm: +res.distance.toFixed(2) });
    }
  }
  drawMeasureOverlay();
  renderMeasureList();
}

/* Measurements are stored in mm; convert only for display. */
function dispMeas(m) {
  const u = $("mUnits").value;
  return { ...m, units: u,
           distance: +(m.distance_mm / UNIT_TO_MM[u]).toPrecision(5) };
}

["mCols", "mRows", "mSquareSize", "mMarkerSize"].forEach((id) =>
  $(id).addEventListener("change", () => {
    clearRect();   // board geometry changed; the warps are no longer valid
    if (S.slug) LS.set(`cvcal:mboard:${S.slug}`, measureSettings());
    if (S.snap) {
      S.measurePts = [];
      prepareBoard();
      remeasureAll();
    }
  }));

$("mUnits").addEventListener("change", () => {
  // display-only: stored mm values just get re-rendered
  if (S.slug) LS.set(`cvcal:mboard:${S.slug}`, measureSettings());
  drawMeasureOverlay();
  renderMeasureList();
  if (R) { R.units = $("mUnits").value; selectRectView(R.sel); }
});

$("snapViewBtn").addEventListener("click", () => {
  if (S.snap) openLightbox(compositeDataURL());
});
$("snapMeasureBtn").addEventListener("click", () => setMeasuring(!S.measuring));

function setMeasuring(on) {
  S.measuring = on;
  S.measurePts = [];
  $("snapMeasureBtn").classList.toggle("active-mode", on);
  $("snapWrap").classList.toggle("measuring", on);
  $("measureHint").textContent = on ? "Click two points on the board plane…" : "";
  drawMeasureOverlay();
}

$("snapCanvas").addEventListener("click", (e) => {
  if (!S.measuring || !S.snap) return;
  const c = $("snapCanvas");
  const rect = c.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (c.width / rect.width);
  const y = (e.clientY - rect.top) * (c.height / rect.height);
  S.measurePts.push([x, y]);
  drawMeasureOverlay();
  if (S.measurePts.length === 2) {
    const [p1, p2] = S.measurePts;
    S.measurePts = [];
    const res = JSON.parse(py.measure_points(p1[0], p1[1], p2[0], p2[1]));
    if (res.error) { toast(res.error, true); drawMeasureOverlay(); return; }
    S.measurements.push({ p1, p2, distance_mm: +res.distance.toFixed(2) });
    drawMeasureOverlay();
    renderMeasureList();
    const last = dispMeas(S.measurements.at(-1));
    $("measureHint").textContent =
      `${last.distance} ${last.units} — click two more points, or toggle 📏 to finish.`;
  }
});

const fmtAlt = (m) => (m.units === "in" || m.units === "ft")
  ? `${m.distance_mm.toFixed(1)} mm`
  : `${(m.distance_mm / 25.4).toFixed(3)} in`;

function renderMeasureList() {
  $("measureList").innerHTML = S.measurements.map(dispMeas).map((m, i) =>
    `#${i + 1}: <b>${m.distance} ${m.units}</b> <span class="dim">(${fmtAlt(m)})</span>`
  ).join(" &nbsp;·&nbsp; ") + (S.measurements.length
    ? ' &nbsp; <a href="#" id="clearMeasures">clear measurements</a>' : "");
  const a = $("clearMeasures");
  if (a) a.addEventListener("click", (e) => {
    e.preventDefault();
    S.measurements = [];
    S.measurePts = [];
    drawMeasureOverlay();
    renderMeasureList();
  });
}

function drawOverlay(canvas, measurements, pts, boardDots) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const s = Math.max(canvas.width / 1280, 0.7);
  if (boardDots) {
    ctx.fillStyle = "rgba(80,200,120,0.6)";
    for (const [x, y] of boardDots) {
      ctx.beginPath(); ctx.arc(x, y, 2.5 * s, 0, Math.PI * 2); ctx.fill();
    }
  }
  const drawPt = (p, color = "#ff5a00") => {
    ctx.beginPath(); ctx.arc(p[0], p[1], 6 * s, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5 * s; ctx.strokeStyle = "#fff"; ctx.stroke();
  };
  for (const m of measurements) {
    ctx.beginPath();
    ctx.moveTo(m.p1[0], m.p1[1]); ctx.lineTo(m.p2[0], m.p2[1]);
    ctx.strokeStyle = "#ffee33"; ctx.lineWidth = 2 * s; ctx.stroke();
    drawPt(m.p1); drawPt(m.p2);
    const mid = [(m.p1[0] + m.p2[0]) / 2, (m.p1[1] + m.p2[1]) / 2 - 12 * s];
    ctx.font = `${Math.round(20 * s)}px system-ui`;
    ctx.textAlign = "center";
    const label = `${m.distance} ${m.units}`;
    ctx.lineWidth = 4 * s; ctx.strokeStyle = "#000";
    ctx.strokeText(label, mid[0], mid[1]);
    ctx.fillStyle = "#ffee33"; ctx.fillText(label, mid[0], mid[1]);
  }
  for (const p of pts) drawPt(p, "#4f9cf7");
}

function drawMeasureOverlay() {
  drawOverlay($("snapCanvas"), S.measurements.map(dispMeas), S.measurePts,
              S.measuring ? S.snapBoard : null);
}

function compositeDataURL() {
  const c = document.createElement("canvas");
  c.width = S.snap.w; c.height = S.snap.h;
  const ctx = c.getContext("2d");
  ctx.drawImage($("snapImg"), 0, 0);
  ctx.drawImage($("snapCanvas"), 0, 0);
  return c.toDataURL("image/png");
}

$("snapSaveBtn").addEventListener("click", () => {
  if (!S.snap) return;
  const a = document.createElement("a");
  a.href = compositeDataURL();
  a.download = `${S.slug || "camera"}_${S.snap.source}_${nowStamp()}.png`;
  a.click();
});

$("snapClearBtn").addEventListener("click", () => {
  S.snap = null;
  S.snapBoard = null;
  S.measurements = [];
  setMeasuring(false);
  clearRect();
  $("snapPanel").style.display = "none";
});

/* --------------------------------------------- orthogonal (top-down) views */
/* R.views[i]: {imgData, width, height, px_per_unit, lo, ratio, warning, ...}
   Measurements are stored in BOARD coordinates (units of R.units), so they
   carry across zoom levels; distances are recomputed from each view's own
   pixels on display. */
let R = null;

const boardToView = (v, b) =>
  [(b[0] - v.lo[0]) * v.px_per_unit, (b[1] - v.lo[1]) * v.px_per_unit];
const viewToBoard = (v, p) =>
  [v.lo[0] + p[0] / v.px_per_unit, v.lo[1] + p[1] / v.px_per_unit];

$("genOrthoBtn").addEventListener("click", () => {
  if (!S.snap || !S.pyReady) return;
  const m = measureSettings();
  const meta = JSON.parse(py.rectify_views(
    S.snap.imageData.data, S.snap.w, S.snap.h,
    m.sx, m.sy, m.square, m.marker, S.snap.source === "undistorted"));
  if (!meta) { toast("Could not rectify — board not found.", true); return; }
  const views = meta.views.map((v, i) => {
    const proxy = py.get_rect_pixels(i);
    const u8 = proxy.toJs();
    proxy.destroy?.();
    return { ...v, imgData: new ImageData(
      new Uint8ClampedArray(u8.buffer || u8), v.width, v.height) };
  });
  R = { views, sel: 0, units: m.units, measuring: false, pts: [],
        measurements: [], undistorted: meta.undistorted ||
                                        S.snap.source === "undistorted" };
  $("rectLabel").textContent =
    `Orthogonal views — board plane is metrically square` +
    (R.undistorted ? "" : " — no calibration: lens distortion not corrected");
  renderRectThumbs();
  selectRectView(0);
  $("rectPanel").style.display = "";
  $("rectPanel").scrollIntoView({ behavior: "smooth", block: "start" });
});

function renderRectThumbs() {
  const wrap = $("rectThumbs");
  wrap.innerHTML = "";
  R.views.forEach((v, i) => {
    const div = document.createElement("div");
    div.className = "rect-thumb" + (i === R.sel ? " sel" : "");
    const c = document.createElement("canvas");
    const tw = 190, th = Math.round(tw * v.height / v.width);
    c.width = tw; c.height = th;
    const src = document.createElement("canvas");
    src.width = v.width; src.height = v.height;
    src.getContext("2d").putImageData(v.imgData, 0, 0);
    c.getContext("2d").drawImage(src, 0, 0, tw, th);
    div.appendChild(c);
    const cap = document.createElement("div");
    cap.className = "cap";
    cap.textContent = v.name;
    div.appendChild(cap);
    if (v.warning || v.clipped) {
      const warn = document.createElement("div");
      warn.className = "warn";
      warn.textContent = (v.clipped ? "⚠ clipped near horizon; " : "⚠ ") +
        "measurements may lose accuracy";
      div.appendChild(warn);
    }
    div.addEventListener("click", () => selectRectView(i));
    wrap.appendChild(div);
  });
}

function selectRectView(i) {
  R.sel = i;
  R.pts = [];
  const v = R.views[i];
  const c = $("rectImg");
  c.width = v.width; c.height = v.height;
  c.getContext("2d").putImageData(v.imgData, 0, 0);
  const oc = $("rectCanvas");
  oc.width = v.width; oc.height = v.height;
  $("rectViewLabel").textContent =
    `${v.name} — ${v.width}×${v.height} — ${v.px_per_unit.toFixed(3)} px/mm`;
  document.querySelectorAll("#rectThumbs .rect-thumb").forEach((el, j) =>
    el.classList.toggle("sel", j === i));
  drawRectOverlay();
  renderRectMeasureList();
}

function clearRect() {
  R = null;
  $("rectPanel").style.display = "none";
  $("rectThumbs").innerHTML = "";
  $("rectWrap").classList.remove("measuring");
  $("rectMeasureBtn").classList.remove("active-mode");
}

function setRectMeasuring(on) {
  if (!R && on) return;
  if (R) { R.measuring = on; R.pts = []; }
  $("rectMeasureBtn").classList.toggle("active-mode", on);
  $("rectWrap").classList.toggle("measuring", on);
  $("rectHint").textContent = on ? "Click two points…" : "";
  drawRectOverlay();
}

/* Distances are recomputed from the CURRENT view's pixel positions, so you
   can compare how they hold up across zoom levels. */
function rectDisplayMeasurements() {
  if (!R) return [];
  const v = R.views[R.sel];        // px_per_unit is px per mm (board units)
  return R.measurements.map((m) => {
    const p1 = boardToView(v, m.b1);
    const p2 = boardToView(v, m.b2);
    const mm = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / v.px_per_unit;
    return { p1, p2, units: R.units, distance_mm: +mm.toFixed(2),
             distance: +(mm / UNIT_TO_MM[R.units]).toPrecision(5) };
  });
}

function drawRectOverlay() {
  drawOverlay($("rectCanvas"), rectDisplayMeasurements(), R ? R.pts : [], null);
}

$("rectCanvas").addEventListener("click", (e) => {
  if (!R || !R.measuring) return;
  const c = $("rectCanvas");
  const rect = c.getBoundingClientRect();
  R.pts.push([(e.clientX - rect.left) * (c.width / rect.width),
              (e.clientY - rect.top) * (c.height / rect.height)]);
  if (R.pts.length === 2) {
    const v = R.views[R.sel];
    const [p1, p2] = R.pts;
    R.pts = [];
    R.measurements.push({ b1: viewToBoard(v, p1), b2: viewToBoard(v, p2) });
    renderRectMeasureList();
    const last = rectDisplayMeasurements().at(-1);
    $("rectHint").textContent =
      `${last.distance} ${R.units} — click two more points, or toggle 📏 to finish.`;
  }
  drawRectOverlay();
});

function remeasureRect() {
  if (!R) return;
  drawRectOverlay();
  renderRectMeasureList();
}

function renderRectMeasureList() {
  const ms = rectDisplayMeasurements();
  $("rectMeasureList").innerHTML = ms.map((m, i) =>
    `#${i + 1}: <b>${m.distance} ${m.units}</b> <span class="dim">(${fmtAlt(m)})</span>`
  ).join(" &nbsp;·&nbsp; ") + (ms.length
    ? ' &nbsp; <a href="#" id="clearRectMeasures">clear measurements</a>' : "");
  const a = $("clearRectMeasures");
  if (a) a.addEventListener("click", (e) => {
    e.preventDefault();
    R.measurements = [];
    R.pts = [];
    drawRectOverlay();
    renderRectMeasureList();
  });
}

function compositeRectDataURL() {
  const v = R.views[R.sel];
  const c = document.createElement("canvas");
  c.width = v.width; c.height = v.height;
  const ctx = c.getContext("2d");
  ctx.drawImage($("rectImg"), 0, 0);
  ctx.drawImage($("rectCanvas"), 0, 0);
  return c.toDataURL("image/png");
}

$("rectViewBtn").addEventListener("click", () => {
  if (R) openLightbox(compositeRectDataURL());
});
$("rectMeasureBtn").addEventListener("click", () => R && setRectMeasuring(!R.measuring));
$("rectSaveBtn").addEventListener("click", () => {
  if (!R) return;
  const name = R.views[R.sel].name.replace(/[^a-z0-9]+/gi, "-");
  const a = document.createElement("a");
  a.href = compositeRectDataURL();
  a.download = `${S.slug || "camera"}_ortho_${name}_${nowStamp()}.png`;
  a.click();
});
$("rectClearBtn").addEventListener("click", clearRect);

/* ------------------------------------------------------- cameras (home) tab */
const CFG = { rows: new Map(), calCache: new Map(), cals: [], camsSig: "",
              abort: null, rendering: false, keepNode: null };

const portKey = (cam) => `cvcal:portlabel:${cam.usb?.bus_path || cam.node}`;
/* How a camera is physically mounted belongs to the PORT, not to the
   calibration file. Stored per-calibration it follows the file: fix a
   camera's rotation while it has the wrong calibration assigned, then fix
   the assignment, and the rotation flips back because it was written into
   the other unit's file. The calibration keeps its own copy for
   downstream consumers, but the port wins for anything on screen. */
const portRotKey = (cam) => `cvcal:portrot:${cam.usb?.bus_path || cam.node}`;
const camRotation = (cam, cal) => {
  const byPort = cam ? rigGet(portRotKey(cam), undefined) : undefined;
  if (byPort != null) return byPort;
  return cal?.extrinsic?.orientation?.rotate_deg_cw ?? null;
};

/* Rig settings live on the host, not in the browser. Which calibration is
   plugged into which USB port is a fact about the machine; kept in
   localStorage it vanishes the moment you drive the tool from a different
   computer, and the fallback silently assigns every camera to the
   alphabetically first calibration. localStorage stays as a mirror so
   browser mode (no bridge) still remembers. */
const RIG = { data: {}, loaded: false };

async function rigLoad() {
  if (!HOST) return;
  try {
    const r = await fetch("api/host/rig");
    if (r.ok) RIG.data = (await r.json()) || {};
  } catch { /* bridge away: fall back to localStorage */ }
  RIG.loaded = true;
  // First run against a host that has never stored rig settings: lift the
  // assignments this browser already holds, so the machine that did the
  // original setup hands them over instead of everyone starting blank.
  if (!Object.keys(RIG.data).length) {
    const carry = {};
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || !/^cvcal:(portlabel|portrot|trackres|trackoff|liveoff):/.test(k)) continue;
      const v = LS.get(k, undefined);
      if (v !== undefined) carry[k] = v;
    }
    if (Object.keys(carry).length) {
      RIG.data = carry;
      try {
        await fetch("api/host/rig", {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(carry) });
        toast(`Moved ${Object.keys(carry).length} camera assignment(s) to ` +
              "the host — they now follow the machine, not this browser.");
      } catch { /* leave them local; nothing is lost */ }
    }
  }
}

function rigGet(key, dflt) {
  if (HOST && RIG.loaded && Object.prototype.hasOwnProperty.call(RIG.data, key)) {
    return RIG.data[key];
  }
  return LS.get(key, dflt);
}

function rigSet(key, val) {
  LS.set(key, val);
  if (!HOST) return;
  RIG.data[key] = val;
  fetch("api/host/rig", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ [key]: val }) }).catch(() => {});
}
const labelToSlug = (base, label) => (label ? `${base}__L${label}` : base);
const displayLabel = (label) => (label ? label : "Default");
function normLabel(v) {
  const c = cleanLabel(v);
  return c.toLowerCase() === "default" ? "" : c;
}

function familyOf(base, cals) {
  const fam = [];
  for (const e of cals) {
    if (e.slug === base) fam.unshift({ label: null, slug: e.slug });
    else if (e.slug.startsWith(base + "__L")) {
      fam.push({ label: e.slug.slice(base.length + 3), slug: e.slug });
    }
  }
  return fam;
}

async function fetchCal(slug) {
  if (CFG.calCache.has(slug)) return CFG.calCache.get(slug);
  let cal = null;
  try {
    const r = await fetch(`api/host/calibrations/${slug}`);
    if (r.ok) cal = await r.json();
  } catch { /* store unreachable */ }
  CFG.calCache.set(slug, cal);
  return cal;
}

const calValid = (cal) => !!cal?.intrinsic?.camera_matrix;

/* --- modal (rename / delete / create) --- */
function modalDialog({ title, body = "", input = null, okText = "OK" }) {
  return new Promise((resolve) => {
    const dlg = $("modal");
    $("modalTitle").textContent = title;
    $("modalBody").innerHTML = body;
    $("modalInputLabel").classList.toggle("hidden", input === null);
    $("modalInput").value = input ?? "";
    $("modalOk").textContent = okText;
    let settled = false;
    const done = (val) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (dlg.open) dlg.close();
      resolve(val);
    };
    const onOk = () => done(input === null ? true : $("modalInput").value);
    const onCancel = () => done(null);
    const onKey = (e) => {
      if (e.key === "Enter") { e.preventDefault(); onOk(); }
    };
    function cleanup() {
      $("modalOk").removeEventListener("click", onOk);
      $("modalCancel").removeEventListener("click", onCancel);
      dlg.removeEventListener("cancel", onCancel);
      $("modalInput").removeEventListener("keydown", onKey);
    }
    $("modalOk").addEventListener("click", onOk);
    $("modalCancel").addEventListener("click", onCancel);
    dlg.addEventListener("cancel", onCancel);
    $("modalInput").addEventListener("keydown", onKey);
    dlg.showModal();
    if (input !== null) { $("modalInput").focus(); $("modalInput").select(); }
  });
}

/* --- render --- */
const camsSignature = (cams) =>
  JSON.stringify(cams.map((c) => [c.node, c.slug, c.path]));

async function renderConfigTab() {
  if (!HOST || CFG.rendering) return;
  CFG.rendering = true;
  try {
    renderStorage();
    CFG.calCache.clear();
    let cams = [], cals = [];
    try { cams = await (await fetch("api/host/cameras")).json(); } catch {}
    try {
      const r = await fetch("api/host/calibrations");
      if (r.ok) cals = await r.json();
    } catch {}
    if (!Array.isArray(cams)) cams = [];
    if (!Array.isArray(cals)) cals = [];
    CFG.camsSig = camsSignature(cams);
    CFG.cals = cals;
    renderCamRows(cams, cals);
    renderCalFiles(cals);
    refreshUsb();              // topology panels (no-op if unchanged)
    await startGridStreams(cams);
  } finally {
    CFG.rendering = false;
  }
}

function renderCamRows(cams, cals) {
  const tbody = $("configRows");
  tbody.innerHTML = "";
  CFG.rows.clear();
  for (const cam of cams) {
    const fam = familyOf(cam.slug, cals);
    const remembered = rigGet(portKey(cam), undefined);
    let sel = fam[0] || null;
    if (remembered !== undefined) {
      const f = fam.find((x) => (x.label || "") === remembered);
      if (f) sel = f;
    }
    const tr = document.createElement("tr");
    const famOpts =
      (fam.length ? "" :
        '<option value="" disabled selected>— not calibrated —</option>') +
      fam.map((f) =>
        `<option value="${esc(f.label || "")}"${f === sel ? " selected" : ""}>` +
        `${esc(displayLabel(f.label))}</option>`).join("") +
      '<option value="__new">➕ Create new calibration…</option>';
    tr.innerHTML = `
      <td><a class="camlink">${esc(cam.name)}</a>
          <div class="dim small">/dev/video${cam.node}</div></td>
      <td>${esc(cam.usb?.id_vendor || "?")}:${esc(cam.usb?.id_product || "?")}
          <div class="dim small">serial ${esc(cam.usb?.serial || "none")}</div>
          ${cam.duplicate
            ? '<div><span class="badge warn dupBadge">duplicate ID — use labels</span></div>'
            : cam.serial_trusted ? "" : '<div><span class="badge warn dupBadge">generic serial</span></div>'}</td>
      <td class="cal-cell">
        <select class="verSel">${famOpts}</select>
        <div class="calinfo dim small"></div>
        <button class="btn small calibBtn">Calibrate</button>
        <div class="conflictNote small hidden">⚠ same calibration selected on several cameras — give each its own label</div>
      </td>
      <td class="live-cell">
          <div class="grid-live-wrap">
            <img class="grid-live" alt="">
            <button class="rotBtn"
              title="Rotate this camera's view 90° clockwise — always saved (with its calibration when one is selected)">⟳</button>
          </div>
          <div class="thumb-note"><span class="spin">◐</span> connecting…</div></td>`;
    tr.querySelector(".camlink").addEventListener("click", () =>
      gotoCollect(cam));
    tr.querySelector(".rotBtn").addEventListener("click", () => rotateCam(row));
    const row = { cam, fam, sel, tr, gotFrame: false };
    const verSel = tr.querySelector(".verSel");
    verSel.addEventListener("change", async () => {
      if (verSel.value === "__new") {
        verSel.value = row.sel ? (row.sel.label || "") : "";
        await createCalFlow(row);
        return;
      }
      row.sel = row.fam.find((x) => (x.label || "") === verSel.value) || null;
      rigSet(portKey(cam), verSel.value);
      updateCalCell(row);
      renderCalFiles(CFG.cals);   // Device column follows the association
    });
    tr.querySelector(".calibBtn").addEventListener("click", () =>
      gotoCalibrate(cam, row.sel ? row.sel.label || "" : ""));
    tbody.appendChild(tr);
    CFG.rows.set(cam.node, row);
    updateCalCell(row);
  }
  $("configNote").textContent = cams.length ? "" : "No cameras detected.";
}

async function updateCalCell(row) {
  const info = row.tr.querySelector(".calinfo");
  const btn = row.tr.querySelector(".calibBtn");
  const img = row.tr.querySelector("img.grid-live");
  // the camera's saved view rotation applies even before any calibration;
  // labeled versions get their own key (clone units can differ)
  const lsRot = LS.get(`cvcal:orient:${row.sel?.slug || row.cam.slug}`,
      LS.get(`cvcal:orient:${row.cam.slug}`, {})).rotate || 0;
  if (!row.sel) {
    info.innerHTML = '<span class="badge warn">not calibrated</span>';
    btn.textContent = "Calibrate";
    row.calState = "none";
    row.rot = lsRot;
  } else {
    const cal = await fetchCal(row.sel.slug);
    if (calValid(cal)) {
      const i = cal.intrinsic;
      info.textContent = `RMS ${i.rms_reprojection_error_px?.toFixed(3)} px ` +
                         `@ ${i.image_size?.join("×")}`;
      btn.textContent = "Recalibrate";
      row.calState = "ok";
    } else {
      info.innerHTML = '<span class="badge warn">uncalibrated</span> — no data yet';
      btn.textContent = "Calibrate";
      row.calState = "uncalibrated";
    }
    if (!row.rotTouched) {      // never clobber a user's fresh ⟳ click
      row.rot = camRotation(row.cam, cal) ?? lsRot;
    }
  }
  applyGridRotation(img, row.rot);
  updateRowHighlights();
}

/* Row status at a glance: red Calibrate = work to do; yellow Recalibrate =
   calibrated but several cameras point at the SAME file (labels needed);
   muted blue-gray Recalibrate = done. When a duplicate-ID camera is done,
   its identity warning fades too — nothing on a finished row shouts. */
function updateRowHighlights() {
  const counts = new Map();
  for (const row of CFG.rows.values()) {
    if (row.sel) counts.set(row.sel.slug, (counts.get(row.sel.slug) || 0) + 1);
  }
  for (const row of CFG.rows.values()) {
    const btn = row.tr.querySelector(".calibBtn");
    const dup = row.tr.querySelector(".dupBadge");
    const note = row.tr.querySelector(".conflictNote");
    const conflict = !!row.sel && counts.get(row.sel.slug) > 1;
    const done = row.calState === "ok" && !conflict;
    btn.classList.toggle("danger", row.calState !== "ok");
    btn.classList.toggle("conflict", row.calState === "ok" && conflict);
    btn.classList.toggle("recal", done);
    note.classList.toggle("hidden", !(row.calState === "ok" && conflict));
    if (dup) {
      dup.classList.toggle("no", done);
      dup.classList.toggle("warn", !done);
    }
  }
}

/* Rotation is cheap to change and expensive to lose: every adjustment is
   saved — to the browser's per-camera key always, into the selected
   calibration file when there is one, and mirrored to the collect tab if
   this camera is selected there. */
async function rotateCam(row) {
  // base on the stored value, not row.rot: a click can land before the
  // row's async calibration fetch has populated it
  const cal = row.sel ? await fetchCal(row.sel.slug) : null;
  const lsRot = LS.get(`cvcal:orient:${row.sel?.slug || row.cam.slug}`,
      LS.get(`cvcal:orient:${row.cam.slug}`, {})).rotate;
  const base = row.rot ?? camRotation(row.cam, cal) ?? lsRot ?? 0;
  row.rot = (base + 90) % 360;
  row.rotTouched = true;
  rigSet(portRotKey(row.cam), row.rot);   // follows the camera, not the file
  applyGridRotation(row.tr.querySelector("img.grid-live"), row.rot);
  LS.set(`cvcal:orient:${row.sel?.slug || row.cam.slug}`, { rotate: row.rot });
  // sync collect/measure ONLY for this exact device — clone units share a
  // slug, so matching by slug would let one unit stomp the other's view
  if (HOST && S.hostCam?.node === row.cam.node) {
    S.orient.rotate = row.rot;
    updateOrientationUI();
    applyOrientationCss();
  }
  if (!row.sel) {
    toast(`Rotation ${row.rot}° saved for this camera (no calibration file yet).`);
    return;
  }
  if (!cal) {
    toast(`Rotation ${row.rot}° saved locally — calibration storage unreachable.`, true);
    return;
  }
  cal.extrinsic = { ...(cal.extrinsic || {}), orientation: {
    rotate_deg_cw: row.rot,
    note: "rotate raw frames clockwise by this before undistortion; " +
          "adjust downstream if the operational mounting differs",
  } };
  CFG.calCache.set(row.sel.slug, cal);
  const cached = LS.cal(row.cam.slug);
  if (cached?.slug === cal.slug) LS.setCal(row.cam.slug, cal);
  if (S.activeCal?.slug === cal.slug) S.activeCal.extrinsic = cal.extrinsic;
  const r = await fetch(`api/host/calibrations/${row.sel.slug}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cal) });
  const res = await r.json().catch(() => ({}));
  if (r.ok && !res.error) {
    toast(`Rotation ${row.rot}° saved to “${displayLabel(row.sel.label)}”.`);
  } else {
    toast("Rotation saved locally, but storage update failed: " +
          (res.error || r.statusText), true);
  }
}

function applyGridRotation(img, rot) {
  if (!rot) { img.style.transform = ""; return; }
  const apply = () => {
    const k = rot % 180 !== 0 && img.naturalWidth
      ? img.naturalHeight / img.naturalWidth : 1;
    img.style.transform = `rotate(${rot}deg) scale(${k})`;
  };
  if (img.naturalWidth) apply();
  else img.addEventListener("load", apply, { once: true });
}

function gotoCollect(cam) {
  CFG.keepNode = cam.node;
  switchTab("collect");
  $("cameraSelect").value = "host:" + cam.node;
  selectCamera("host:" + cam.node).finally(() => { CFG.keepNode = null; });
}

function gotoCalibrate(cam, label) {
  rigSet(portKey(cam), label || "");
  LS.set(`cvcal:name:${cam.slug}`, label || "");
  gotoCollect(cam);
}

/* --- create a placeholder ("null") calibration file --- */
async function createCalFlow(row) {
  const val = await modalDialog({
    title: "New calibration file",
    body: `Creates an empty calibration entry for <b>${esc(row.cam.name)}</b> ` +
      `(<code>${esc(row.cam.slug)}</code>). If you have several identical ` +
      `cameras, give each a short label and write it on the camera body; ` +
      `otherwise keep “Default”.`,
    input: "Default", okText: "Create" });
  if (val === null) return;
  const label = normLabel(val);
  const slug = labelToSlug(row.cam.slug, label);
  if (CFG.cals.some((c) => c.slug === slug)) {
    toast(`“${displayLabel(label)}” already exists for this camera — select it instead.`, true);
    return;
  }
  const cam = row.cam;
  const placeholder = {
    schema_version: 1,
    name: label || cam.name,
    slug,
    camera: { platform: "host-bridge", label: cam.name, usb: cam.usb,
              serial_trusted: !!cam.serial_trusted, by_id: cam.by_id || null,
              assigned_label: label || null,
              serial_generic: !cam.serial_trusted },
    intrinsic: null,
    uncalibrated: true,
    extrinsic: {},
  };
  const r = await fetch(`api/host/calibrations/${slug}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(placeholder) });
  const res = await r.json();
  if (!r.ok || res.error) {
    toast("Could not create file: " + (res.error || r.statusText), true);
    return;
  }
  rigSet(portKey(cam), label);
  toast(`Created “${displayLabel(label)}” — click Calibrate to collect images.`);
  renderConfigTab();
}

/* --- calibration files panel (right column) --- */
function devicesFor(slug) {
  const out = [];
  for (const row of CFG.rows.values()) {
    if (row.sel?.slug === slug) out.push(`/dev/video${row.cam.node}`);
  }
  return out;
}

async function renderCalFiles(cals) {
  // the non-selected backend (dir<->S3), for the copy-across buttons
  try { CFG.alt = await (await fetch("api/host/storage/alt")).json(); }
  catch { CFG.alt = { ok: false }; }
  const altOk = !!CFG.alt?.ok;
  const altSlugs = new Set(CFG.alt?.slugs || []);
  const altName = CFG.alt?.type === "dir"
    ? `local directory (${CFG.alt.path || "camera_cal"})` : "S3";
  const altShort = CFG.alt?.type === "dir" ? "local dir" : "S3";
  const box = $("calFiles");
  box.innerHTML = "";
  $("calFilesNote").textContent = cals.length
    ? "" : "No calibration files in the selected storage yet.";
  // group versions by base camera ID
  const groups = new Map();
  for (const e of cals) {
    const base = e.slug.split("__L")[0];
    if (!groups.has(base)) groups.set(base, []);
  }
  for (const [base, vers] of groups) {
    for (const f of familyOf(base, cals)) vers.push(f);
  }
  for (const [base, vers] of groups) {
    if (!vers.length) continue;
    const div = document.createElement("div");
    div.className = "cal-group";
    div.innerHTML = `<div class="cal-group-id"><code>${esc(base)}</code></div>
      <table class="calfam-table">
        <thead><tr><th>Label</th><th>Device</th><th>Status</th><th></th></tr></thead>
        <tbody></tbody>
      </table>`;
    const tbody = div.querySelector("tbody");
    for (const ver of vers) {
      const tr = document.createElement("tr");
      const devs = devicesFor(ver.slug);
      const inAlt = altOk && altSlugs.has(ver.slug);
      const copyTitle = altOk
        ? `Copy to ${altName}` + (inAlt ? " (updates the existing copy)" : "")
        : `Other storage unavailable — ${CFG.alt?.error ||
           "configure it in Calibration Storage"}`;
      tr.innerHTML = `
        <td><b>${esc(displayLabel(ver.label))}</b></td>
        <td>${devs.length ? esc(devs.join(", ")) : '<span class="dim">—</span>'}</td>
        <td class="ver-status dim small">…</td>
        <td class="ver-acts">
          <button class="btn small" title="Rename label">✎</button>
          <button class="btn small copyBtn" title="${esc(copyTitle)}"${altOk ? "" : " disabled"}>⧉</button>
          <button class="btn small danger-outline" title="Delete calibration file">🗑</button>
        </td>`;
      const [renameBtn, copyBtn, delBtn] = tr.querySelectorAll("button");
      renameBtn.addEventListener("click", () => renameCalFlow(base, ver));
      delBtn.addEventListener("click", () => deleteCalFlow(base, ver));
      copyBtn.addEventListener("click", async () => {
        copyBtn.disabled = true;
        const r = await fetch(`api/host/calibrations/${ver.slug}/copy_alt`,
                              { method: "POST" });
        const res = await r.json().catch(() => ({}));
        if (r.ok && !res.error) {
          toast(`Copied “${displayLabel(ver.label)}” to ${altShort} — ${res.location || ""}`);
          renderCalFiles(CFG.cals);      // refresh the presence markers
        } else {
          toast("Copy failed: " + (res.error || r.statusText), true);
          copyBtn.disabled = false;
        }
      });
      fetchCal(ver.slug).then((cal) => {
        const st = tr.querySelector(".ver-status");
        st.innerHTML = (calValid(cal)
          ? `RMS ${cal.intrinsic.rms_reprojection_error_px?.toFixed(3)} px`
          : '<span class="badge warn">uncalibrated</span>') +
          (inAlt ? ` <span class="dim">· ✓ ${esc(altShort)}</span>` : "");
      });
      tbody.appendChild(tr);
    }
    box.appendChild(div);
  }
}

async function renameCalFlow(base, ver) {
  const val = await modalDialog({
    title: `Rename “${displayLabel(ver.label)}”`,
    body: `Camera ID <code>${esc(base)}</code>. Enter a new label, or ` +
      `“Default” for the unlabeled slot. Cameras linked to this label ` +
      `follow the rename.`,
    input: displayLabel(ver.label), okText: "Rename" });
  if (val === null) return;
  const label = normLabel(val);
  if (label === (ver.label || "")) return;
  const r = await fetch(`api/host/calibrations/${ver.slug}/rename`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_slug: labelToSlug(base, label), label }) });
  const res = await r.json();
  if (!r.ok || res.error) {
    toast(res.error || "Rename failed.", true);
    return;
  }
  for (const row of CFG.rows.values()) {
    if (row.sel?.slug === ver.slug) rigSet(portKey(row.cam), label);
  }
  toast(`Renamed to “${displayLabel(label)}”.`);
  renderConfigTab();
}

async function deleteCalFlow(base, ver) {
  const devs = devicesFor(ver.slug);
  const ok = await modalDialog({
    title: `Delete “${displayLabel(ver.label)}”?`,
    body: `<code>${esc(ver.slug)}</code> will be removed from storage. ` +
      `This cannot be undone.` +
      (devs.length
        ? `<br><br>⚠ <b>Currently linked to ${esc(devs.join(", "))}</b> — ` +
          `that camera reverts to Default or “not calibrated”.`
        : ""),
    okText: "Delete" });
  if (!ok) return;
  const r = await fetch(`api/host/calibrations/${ver.slug}`,
                        { method: "DELETE" });
  const res = await r.json().catch(() => ({}));
  if (!r.ok || res.error) {
    toast(res.error || "Delete failed.", true);
    return;
  }
  for (const row of CFG.rows.values()) {
    if (row.sel?.slug === ver.slug) LS.del(portKey(row.cam));
  }
  // drop any stale copy cached in the browser for the measure tab
  if (LS.cal(base)?.slug === ver.slug) LS.delCal(base);
  toast(`Deleted “${displayLabel(ver.label)}”.`);
  renderConfigTab();
}

/* --- live views: all cameras over ONE multiplexed connection (browsers
   allow only ~6 parallel connections per host, so 8 MJPEG <img> streams
   would starve the API). Frames arrive as "<node>,<len>\n" + jpeg. --- */
async function startGridStreams(cams) {
  stopMultiReader();
  if (!cams.length) return;
  let active = {};
  try { active = await (await fetch("api/host/streams")).json(); } catch {}
  await Promise.all(cams.map(async (cam) => {
    if (active[cam.node]) return;   // reuse (e.g. the collect tab's stream)
    try {
      await fetch("api/host/stream/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node: cam.node, width: 640, height: 480,
                               fps: 15 }) });
    } catch { /* busy or unplugged — row just shows no frame */ }
  }));
  openMultiStream(cams.map((c) => c.node));
}

function stopMultiReader() {
  if (CFG.abort) { CFG.abort.abort(); CFG.abort = null; }
}

async function openMultiStream(nodes) {
  const ctrl = new AbortController();
  CFG.abort = ctrl;
  try {
    await readFrameStream(
      `api/host/multistream?nodes=${nodes.join(",")}&width=480&t=${Date.now()}`,
      ctrl, paintGridFrame);
  } catch { /* aborted, or bridge went away */ }
  if (CFG.abort === ctrl) {
    CFG.abort = null;
    if (activeTab === "cameras") {           // reconnect while tab is open
      setTimeout(() => {
        if (activeTab === "cameras" && !CFG.abort) openMultiStream(nodes);
      }, 1500);
    }
  }
}

function paintGridFrame(node, jpg) {
  const row = CFG.rows.get(node);
  const img = row?.tr.querySelector("img.grid-live");
  if (!img) return;
  const url = URL.createObjectURL(new Blob([jpg], { type: "image/jpeg" }));
  const old = img.dataset.blob;
  img.onload = () => { if (old) URL.revokeObjectURL(old); };
  img.dataset.blob = url;
  img.src = url;
  if (!row.gotFrame) {
    row.gotFrame = true;
    row.tr.querySelector(".thumb-note").textContent = "";
    applyGridRotation(img, row.rot || 0);
  }
}

async function stopGridStreams() {
  stopMultiReader();
  const keep = new Set(
    [S.hostCam?.node, CFG.keepNode].filter((n) => n != null));
  try {
    const active = await (await fetch("api/host/streams")).json();
    for (const n of Object.keys(active)) {
      if (!keep.has(+n)) stopHostStream(+n);
    }
  } catch { /* bridge unreachable */ }
}

/* --- background probing: notice plugged/unplugged devices --- */
async function pollCameras() {
  if (!HOST || activeTab !== "cameras" || CFG.rendering || document.hidden ||
      $("modal").open) return;
  refreshUsb();                // no-op unless the USB topology changed
  let cams;
  try { cams = await (await fetch("api/host/cameras")).json(); } catch { return; }
  if (!Array.isArray(cams)) return;
  if (camsSignature(cams) !== CFG.camsSig) {
    toast("Camera list changed — refreshing.");
    refreshDevices();          // keep the collect/measure dropdowns in sync
    renderConfigTab();
  }
}

$("camRefreshBtn").addEventListener("click", () => {
  refreshDevices();
  renderConfigTab();
});

/* --- USB topology view: tree of checked devices grouped by controller.
   Devices on one controller share its bandwidth, so the point of this
   view is seeing whether cameras/mics are spread evenly. --- */
const USB = { devices: [], sig: "", overrides: LS.get("cvcal:usbchecks", {}) };

const usbKey = (d) => `${d.key}|${d.vid || ""}:${d.pid || ""}`;
const usbAutoChecked = (d) =>
  d.is_root || d.is_hub || d.has_video || d.has_audio;
const usbChecked = (d) => {
  const o = USB.overrides[usbKey(d)];
  return o === undefined ? usbAutoChecked(d) : o;
};

function usbSpeed(d) {
  const s = parseFloat(d.speed_mbps);
  if (!s) return "";
  return s >= 1000 ? `${s / 1000} Gbps` : `${s} Mbps`;
}
const usbIcons = (d) =>
  (d.is_root ? "🖥 " : d.is_hub ? "🔀 " : "") +
  (d.has_video ? "🎥 " : "") + (d.has_audio ? "🎤 " : "");
const usbName = (d) => d.is_root
  ? `USB ${d.usb_version || ""} bus ${d.bus}`.replace("  ", " ")
  : (d.product || d.manufacturer || `${d.vid}:${d.pid}`);
const shortCtrl = (c) => (c || "?").split(":").slice(-2).join(":");
const usbCmp = (a, b) =>
  a.key.localeCompare(b.key, undefined, { numeric: true });

async function refreshUsb(force = false) {
  if (!HOST) return;
  let devs;
  try { devs = await (await fetch("api/host/usb")).json(); } catch { return; }
  if (!Array.isArray(devs)) return;
  const sig = JSON.stringify(devs.map((d) => [d.key, d.vid, d.pid]));
  if (!force && sig === USB.sig) return;
  USB.sig = sig;
  USB.devices = devs;
  renderUsbList();
  renderUsbTree();
}

function renderUsbList() {
  const box = $("usbList");
  box.innerHTML = "";
  const devs = [...USB.devices].sort((a, b) =>
    (a.controller || "").localeCompare(b.controller || "") || usbCmp(a, b));
  for (const d of devs) {
    const row = document.createElement("label");
    row.className = "usb-row";
    row.innerHTML = `
      <input type="checkbox"${usbChecked(d) ? " checked" : ""}>
      <span class="path">${esc(d.key)}</span>
      <span class="icons">${usbIcons(d) || "▫"}</span>
      <span class="grow">${esc(usbName(d))}
        ${d.is_root ? "" : `<span class="dim small">${esc(d.vid)}:${esc(d.pid)}</span>`}
        ${d.video_devs?.length ? `<span class="usb-cam">${esc(d.video_devs.join(", "))}</span>` : ""}
      </span>
      <span class="dim small">${shortCtrl(d.controller)}</span>`;
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked === usbAutoChecked(d)) {
        delete USB.overrides[usbKey(d)];      // back to automatic
      } else {
        USB.overrides[usbKey(d)] = e.target.checked;
      }
      LS.set("cvcal:usbchecks", USB.overrides);
      renderUsbTree();
    });
    box.appendChild(row);
  }
}

function renderUsbTree() {
  const box = $("usbTree");
  const devs = USB.devices;
  if (!devs.length) {
    box.innerHTML = '<span class="dim">No USB devices reported.</span>';
    return;
  }
  const byKey = new Map(devs.map((d) => [d.key, d]));
  // checked devices are visible; their ancestors stay visible (dimmed if
  // unchecked) so the topology never has gaps
  const visible = new Set();
  for (const d of devs) {
    if (!usbChecked(d)) continue;
    let cur = d;
    while (cur && !visible.has(cur.key)) {
      visible.add(cur.key);
      cur = cur.parent ? byKey.get(cur.parent) : null;
    }
  }
  const children = new Map();
  for (const d of devs) {
    if (!visible.has(d.key) || d.is_root) continue;
    if (!children.has(d.parent)) children.set(d.parent, []);
    children.get(d.parent).push(d);
  }
  const nodeHtml = (d) => {
    const kids = (children.get(d.key) || []).sort(usbCmp);
    const cams = d.video_devs?.length
      ? ` <span class="usb-cam">${esc(d.video_devs.join(", "))}</span>` : "";
    return `<li><div class="usb-node${usbChecked(d) ? "" : " dimnode"}">` +
      `${usbIcons(d)}<b>${esc(usbName(d))}</b>` +
      (d.is_root ? "" :
        ` <span class="dim small">${esc(d.key)} · ${esc(d.vid)}:${esc(d.pid)}</span>`) +
      (usbSpeed(d) ? ` <span class="usb-speed">${usbSpeed(d)}</span>` : "") +
      cams + `</div>` +
      (kids.length ? `<ul>${kids.map(nodeHtml).join("")}</ul>` : "") + "</li>";
  };
  const ctrls = [...new Set(devs.map((d) => d.controller).filter(Boolean))]
    .sort();
  box.innerHTML = ctrls.map((c) => {
    const sub = devs.filter((d) => d.controller === c);
    const roots = sub.filter((d) => d.is_root && visible.has(d.key))
      .sort(usbCmp);
    if (!roots.length) return "";
    const nv = sub.filter((d) => d.has_video).length;
    const na = sub.filter((d) => d.has_audio).length;
    const load = [nv ? `${nv} video` : "", na ? `${na} audio` : ""]
      .filter(Boolean).join(" · ");
    return `<div class="usb-ctrl">
      <div class="usb-ctrl-head">Controller <code>${esc(c)}</code>
        <span class="badge ${nv + na ? "warn" : "no"}">${load || "no A/V"}</span>
      </div>
      <ul class="usb-root">${roots.map(nodeHtml).join("")}</ul></div>`;
  }).join("");
}

/* ------------------------------------------------------------ tracking tab */
/* Live ArUco tracking across ALL cameras at once. Detection runs on the
   bridge (native OpenCV, multi-core) — the browser only renders frames and
   overlay JSON, so the Load panel measures what a real deployment costs. */
const TR = { on: false, abort: null, poll: null, cams: [], results: {},
             bytes: 0, rateT: 0, rateB: 0, mbps: 0, starting: false,
             restartQueued: false };

function tkEffFps() {
  const view = Math.min(30, Math.max(1, parseFloat($("tkViewFps").value) || 10));
  const want = Math.max(0.2, parseFloat($("tkTrackFps").value) || 5);
  // tracker runs every Nth view frame; round silently to the closest fit
  const every = Math.max(1, Math.round(view / Math.min(want, view)));
  return { view, every, track: +(view / every).toFixed(2) };
}

function tkShowEff() {
  const { view, every, track } = tkEffFps();
  $("tkEff").textContent =
    `detecting on ${every === 1 ? "every frame" : `1 of every ${every} frames`}` +
    ` → ${track} fps` +
    (view / every === parseFloat($("tkTrackFps").value) ? "" : " (rounded)");
  return { view, track };
}

const tkCamKey = (cam) => cam.usb?.bus_path || String(cam.node);

async function tkBuildCams() {
  let cams = [], cals = [], usb = [];
  try { cams = await (await fetch("api/host/cameras")).json(); } catch {}
  try {
    const r = await fetch("api/host/calibrations");
    if (r.ok) cals = await r.json();
  } catch {}
  try { usb = await (await fetch("api/host/usb")).json(); } catch {}
  if (!Array.isArray(cams)) cams = [];
  const usbByKey = new Map((Array.isArray(usb) ? usb : [])
    .map((d) => [d.key, d]));
  TR.cams = [];
  for (const cam of cams) {
    const fam = familyOf(cam.slug, Array.isArray(cals) ? cals : []);
    const rem = rigGet(portKey(cam), undefined);
    let sel = fam[0] || null;
    if (rem !== undefined) {
      const f = fam.find((x) => (x.label || "") === rem);
      if (f) sel = f;
    }
    const cal = sel ? await fetchCal(sel.slug) : null;
    const rot = camRotation(cam, cal) ??
      LS.get(`cvcal:orient:${sel?.slug || cam.slug}`,
             LS.get(`cvcal:orient:${cam.slug}`, {})).rotate ?? 0;
    let modes = [];
    try {
      const det = await (await fetch(
        `api/host/cameras/${cam.node}/details`)).json();
      const seen = new Set();
      modes = (det.modes || [])
        .filter((m) => !seen.has(`${m.width}x${m.height}`) &&
                       seen.add(`${m.width}x${m.height}`))
        .map((m) => [m.width, m.height])
        .sort((a, b) => b[0] * b[1] - a[0] * a[1]);
    } catch {}
    if (!modes.length) modes = [[1280, 720], [640, 480]];
    const savedRes = rigGet(`cvcal:trackres:${tkCamKey(cam)}`, null);
    const res = (savedRes && modes.some(([w, h]) =>
        w === savedRes[0] && h === savedRes[1])) ? savedRes
      : modes.find(([w, h]) => w * h <= 1280 * 720) || modes[modes.length - 1];
    TR.cams.push({
      cam, sel, rot, modes, res,
      calOk: calValid(cal),
      // saved world pose from a previous solve — the default camera pose
      // until pose estimation is run again
      pose: cal?.extrinsic?.world_pose || null,
      ctrl: usbByKey.get(cam.usb?.bus_path)?.controller || "?",
      enabled: !rigGet(`cvcal:trackoff:${tkCamKey(cam)}`, false),
      canvas: null, stat: null, busy: false,
    });
  }
}

function tkRenderCamList() {
  const box = $("tkCamList");
  box.innerHTML = "";
  if (!TR.cams.length) {
    box.textContent = "No cameras detected.";
    return;
  }
  const groups = new Map();
  for (const c of TR.cams) {
    if (!groups.has(c.ctrl)) groups.set(c.ctrl, []);
    groups.get(c.ctrl).push(c);
  }
  for (const [ctrl, list] of [...groups].sort()) {
    const div = document.createElement("div");
    div.className = "tk-ctrl";
    div.innerHTML =
      `<div class="tk-ctrl-head">Controller <code>${esc(shortCtrl(ctrl))}</code></div>`;
    for (const c of list) {
      const row = document.createElement("div");
      row.className = "tk-camrow";
      row.innerHTML = `
        <label class="tk-camlabel"><input type="checkbox"${c.enabled ? " checked" : ""}>
          <span>${esc(c.sel?.label ? c.sel.label + " — " : "")}${esc(c.cam.name)}
            <span class="dim">/dev/video${c.cam.node}</span></span></label>
        <select class="tk-res" title="Capture resolution for this camera">
          ${c.modes.map(([w, h]) =>
            `<option value="${w}x${h}"${w === c.res[0] && h === c.res[1]
              ? " selected" : ""}>${w}×${h}</option>`).join("")}
        </select>
        <span class="tk-pose ${c.pose ? "has" : "none"}" title="${
          c.pose ? esc("saved " + (c.pose.solved_at || "").slice(0, 16).replace("T", " ") +
                       " · marker " + (c.pose.marker_mm ?? "?") + " mm" +
                       (c.pose.rms_px != null ? " · solve RMS " + c.pose.rms_px + " px" : ""))
                 : "no saved world pose — run pose estimation"
        }">${c.pose ? "◈ posed" : "◇ no pose"}</span>`;
      row.querySelector("input").addEventListener("change", (e) => {
        c.enabled = e.target.checked;
        rigSet(`cvcal:trackoff:${tkCamKey(c.cam)}`, !c.enabled);
        tkRenderViews();
        tkStart();
      });
      row.querySelector(".tk-res").addEventListener("change", (e) => {
        c.res = e.target.value.split("x").map(Number);
        rigSet(`cvcal:trackres:${tkCamKey(c.cam)}`, c.res);
        tkStart();
      });
      div.appendChild(row);
    }
    box.appendChild(div);
  }
}

function tkRenderViews() {
  const box = $("tkViews");
  box.innerHTML = "";
  for (const c of TR.cams) {
    if (!c.enabled) { c.canvas = c.stat = null; continue; }
    const card = document.createElement("div");
    card.className = "tk-card" + (c.calOk ? "" : " uncal");
    card.innerHTML = `
      <div class="tk-cap">${esc(c.sel?.label ? c.sel.label + " — " : "")}${esc(c.cam.name)}
        <span class="dim small">/dev/video${c.cam.node}${c.calOk ? "" : " · uncalibrated"}</span></div>
      <canvas width="16" height="9"></canvas>
      <div class="tk-stat dim small">starting…</div>`;
    box.appendChild(card);
    c.canvas = card.querySelector("canvas");
    c.stat = card.querySelector(".tk-stat");
  }
  if (!box.children.length) {
    box.innerHTML = '<p class="dim">No cameras selected.</p>';
  }
}

async function tkStart() {
  if (!TR.on) return;
  if (TR.starting) { TR.restartQueued = true; return; }
  TR.starting = true;
  try {
    do {
      TR.restartQueued = false;
      const { view, track } = tkShowEff();
      LS.set("cvcal:track", { view: $("tkViewFps").value,
        track: $("tkTrackFps").value, marker: $("tkMarkerMm").value });
      stopTkReader();
      const active = TR.cams.filter((c) => c.enabled);
      await fetch("api/host/track/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          view_fps: view, track_fps: track,
          marker_mm: parseFloat($("tkMarkerMm").value) || 40,
          keep: S.hostCam ? [S.hostCam.node] : [],
          cameras: active.map((c) => ({
            node: c.cam.node, width: c.res[0], height: c.res[1],
            cal_slug: c.sel ? c.sel.slug : null })),
        }) });
      if (active.length) {
        tkOpenStream(active.map((c) => c.cam.node), view);
      }
      tkStartPolling(track);
    } while (TR.restartQueued);
  } catch (e) {
    toast("Could not start tracking: " + e.message, true);
  } finally {
    TR.starting = false;
  }
}

function stopTkReader() {
  if (TR.abort) { TR.abort.abort(); TR.abort = null; }
}

async function tkOpenStream(nodes, fps) {
  const ctrl = new AbortController();
  TR.abort = ctrl;
  while (TR.abort === ctrl && TR.on) {
    try {
      await readFrameStream(
        `api/host/multistream?nodes=${nodes.join(",")}&width=720&quality=80` +
        `&fps=${fps}&t=${Date.now()}`, ctrl, tkFrame);
    } catch { /* aborted or bridge away */ }
    if (TR.abort !== ctrl || !TR.on) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function tkFrame(node, jpg) {
  TR.bytes += jpg.length;
  const c = TR.cams.find((x) => x.cam.node === node && x.enabled);
  if (!c || !c.canvas || c.busy) return;
  c.busy = true;
  try {
    const bmp = await createImageBitmap(new Blob([jpg], { type: "image/jpeg" }));
    tkDraw(c, bmp);
    bmp.close?.();
  } catch { /* partial frame */ } finally {
    c.busy = false;
  }
}

function tkRotPt(x, y, rot, w, h) {
  switch (((rot % 360) + 360) % 360) {
    case 90: return [h - y, x];
    case 180: return [w - x, h - y];
    case 270: return [y, w - x];
    default: return [x, y];
  }
}

function tkDraw(c, bmp) {
  const rot = c.rot || 0, swap = rot % 180 !== 0;
  const W = bmp.width, H = bmp.height;
  const cw = swap ? H : W, ch = swap ? W : H;
  const cv = c.canvas;
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext("2d");
  ctx.save();
  ctx.translate(cw / 2, ch / 2);
  ctx.rotate(rot * Math.PI / 180);
  ctx.drawImage(bmp, -W / 2, -H / 2);
  ctx.restore();
  const res = TR.results[c.cam.node];
  if (!res?.tags?.length) return;
  // detection ran at capture resolution; view frames may be downscaled
  const s = W / (res.size?.[0] || c.res[0]);
  const P = (pt) => tkRotPt(pt[0] * s, pt[1] * s, rot, W, H);
  for (const t of res.tags) {
    const q = t.corners.map(P);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#9aa4b2";                    // far sides: gray
    ctx.beginPath();
    ctx.moveTo(q[1][0], q[1][1]);
    ctx.lineTo(q[2][0], q[2][1]);
    ctx.lineTo(q[3][0], q[3][1]);
    ctx.stroke();
    ctx.strokeStyle = "#4f9cf7";                    // X axis: blue
    ctx.beginPath();
    ctx.moveTo(q[0][0], q[0][1]);
    ctx.lineTo(q[1][0], q[1][1]);
    ctx.stroke();
    ctx.strokeStyle = "#4bd66a";                    // Y axis: green
    ctx.beginPath();
    ctx.moveTo(q[0][0], q[0][1]);
    ctx.lineTo(q[3][0], q[3][1]);
    ctx.stroke();
    ctx.fillStyle = "#ff4fd8";                      // origin corner
    ctx.beginPath();
    ctx.arc(q[0][0], q[0][1], 4, 0, Math.PI * 2);
    ctx.fill();
    const dist = t.distance_mm == null ? "" :
      " · " + (t.approx ? "~" : "") + (t.distance_mm >= 1000
        ? (t.distance_mm / 1000).toFixed(2) + " m"
        : Math.round(t.distance_mm) + " mm");
    const label = `ID ${t.id}${dist}`;
    const lx = Math.min(cw - 8, Math.max(...q.map((p) => p[0])) + 6);
    const ly = Math.max(14, Math.min(...q.map((p) => p[1])) + 12);
    ctx.font = "13px system-ui";
    ctx.textAlign = "left";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#000";
    ctx.strokeText(label, lx, ly);
    ctx.fillStyle = "#ffee33";
    ctx.fillText(label, lx, ly);
  }
}

function tkStartPolling(trackFps) {
  clearInterval(TR.poll);
  TR.poll = setInterval(tkPoll, 1000 / Math.min(10, Math.max(2, trackFps)));
}

async function tkPoll() {
  if (!TR.on || document.hidden) return;
  let snap;
  try {
    snap = await (await fetch("api/host/track/results")).json();
  } catch { return; }
  TR.results = snap.results || {};
  for (const c of TR.cams) {
    if (!c.enabled || !c.stat) continue;
    const r = TR.results[c.cam.node];
    const cap = (snap.capture || {})[c.cam.node];
    if (cap && cap.error) {
      // a stream that died carries its reason — show that instead of
      // leaving the tile on "warming up…" forever
      c.stat.textContent = `⚠ ${cap.error}`;
      c.stat.classList.add("stat-error");
      continue;
    }
    c.stat.classList.remove("stat-error");
    c.stat.textContent =
      (cap ? `capture ${cap.fps}/${cap.target || "?"} fps` : "no stream") +
      (r ? ` · detect ${r.detect_ms} ms · ${r.n} tag${r.n === 1 ? "" : "s"}`
         : " · warming up…");
  }
  tkRenderMetrics(snap);
}

function tkRenderMetrics(snap) {
  const m = snap.metrics || {};
  const st = snap.stats || {};
  const { track } = tkEffFps();
  const now = Date.now();
  if (now - TR.rateT > 2000) {
    TR.mbps = (TR.bytes - TR.rateB) / ((now - TR.rateT) / 1000) / 1048576;
    TR.rateT = now;
    TR.rateB = TR.bytes;
  }
  const rows = [];
  rows.push(["Tracker rate", st.achieved_fps
    ? `${st.achieved_fps} / ${track} fps (${Math.min(999,
        Math.round(100 * st.achieved_fps / track))}%)`
    : (snap.running ? "warming up…" : "stopped")]);
  if (st.duty_pct != null && st.achieved_fps) {
    rows.push(["Tracker duty", `${st.duty_pct}% of the loop busy`]);
  }
  if (m.proc_cpu_pct_one_core != null) {
    rows.push(["CPU util.",
      `${(m.proc_cpu_pct_one_core / 100).toFixed(2)} of ${m.ncpu} cores ` +
      `(${m.proc_cpu_pct_machine}%)`]);
  }
  if (m.rss_mb != null) {
    rows.push(["Bridge memory", `${m.rss_mb} MB` +
      (m.rss_pct != null ? ` (${m.rss_pct}%)` : "")]);
  }
  if (m.loadavg) {
    rows.push(["Load avg", `${m.loadavg.join(" / ")} (${m.ncpu} cores)`]);
  }
  rows.push(["View stream", `${TR.mbps.toFixed(2)} MB/s`]);
  const caps = Object.entries(snap.capture || {}).map(([n, cp]) => {
    const pctv = cp.target ? Math.round(100 * cp.fps / cp.target) : null;
    return `video${n} · ${cp.width}×${cp.height} · ${cp.fps}` +
      (cp.target ? `/${cp.target} fps (${pctv}%)` : " fps");
  });
  $("tkMetrics").innerHTML = rows.map(([k, v]) =>
    `<div class="tk-mrow"><span class="dim">${k}</span><b>${esc(String(v))}</b></div>`
  ).join("") + (caps.length
    ? `<div class="tk-mcap dim">${caps.map(esc).join("<br>")}</div>` : "");
}

async function enterTracking() {
  if (!HOST || TR.on) return;
  TR.on = true;
  const p = LS.get("cvcal:track", null);
  if (p) {
    if (p.view) $("tkViewFps").value = p.view;
    if (p.track) $("tkTrackFps").value = p.track;
    if (p.marker) $("tkMarkerMm").value = p.marker;
  }
  tkShowEff();
  TR.rateT = Date.now();
  TR.rateB = TR.bytes;
  await tkBuildCams();
  if (!TR.on) return;                  // user already left the tab
  tkRenderCamList();
  renderRefCamOptions();
  tkRenderViews();
  if (!W3.data) {
    // empty but real: axes and any camera that already has a saved pose,
    // so the view is never a blank panel waiting on a button
    W3.data = { marker_mm: 40, tags: [], unlinked_cameras: [],
                unlinked_tags: [], root: null, rms_px: null,
                cameras: TR.cams.filter((c) => c.pose?.T_world_cam)
                  .map((c) => ({ node: c.cam.node, T: c.pose.T_world_cam,
                                 pos: c.pose.position_mm || [0, 0, 0],
                                 seen: [] })) };
    w3Fit(W3.data);
  }
  w3Render();
  await tkStart();
}

function leaveTracking() {
  if (!TR.on) return;
  TR.on = false;
  if (WCAL.on) { WCAL.on = false; WCAL.n = 0; updateWcalUI(); }
  clearInterval(TR.poll);
  TR.poll = null;
  stopTkReader();
  const stopNodes = TR.cams
    .filter((c) => c.enabled && c.cam.node !== S.hostCam?.node)
    .map((c) => c.cam.node);
  fetch("api/host/track/stop", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stop_streams: stopNodes }) }).catch(() => {});
}

["tkViewFps", "tkTrackFps", "tkMarkerMm"].forEach((id) =>
  $(id).addEventListener("change", () => {
    if (!TR.on) return;
    if (id === "tkViewFps" || TR.starting) {
      // view-fps needs a stream restart; and if a restart is already in
      // flight, fold the change into it so it can't be overwritten
      tkStart();
    } else {
      const { track } = tkShowEff();
      fetch("api/host/track/config", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_fps: track,
          marker_mm: parseFloat($("tkMarkerMm").value) || 40 }) });
      tkStartPolling(track);
      LS.set("cvcal:track", { view: $("tkViewFps").value,
        track: $("tkTrackFps").value, marker: $("tkMarkerMm").value });
    }
  }));

/* -------------------------------------------- world coordinate system (3D) */
/* Static solve on demand: the bridge grabs the newest frame per camera,
   PnP-solves every visible tag, and chains poses from the world tag. The
   viewer is a small hand-rolled wireframe renderer: orbit (drag), pan
   (right/ctrl-drag), zoom (wheel). World units are mm; the world tag's
   plane is z=0 with +z out of the tag. */
const W3 = { yaw: 0.7, pitch: 0.9, dist: 2000, target: [0, 0, 0],
             data: null, drag: null };

function w3Rot(p) {
  // world -> view rotation (orbit): yaw about world Z, then pitch about X
  const [tx, ty, tz] = W3.target;
  const x = p[0] - tx, y = p[1] - ty, z = p[2] - tz;
  const cy = Math.cos(W3.yaw), sy = Math.sin(W3.yaw);
  const x1 = cy * x + sy * y, y1 = -sy * x + cy * y;
  const cp = Math.cos(W3.pitch), sp = Math.sin(W3.pitch);
  return [x1, cp * y1 + sp * z, -sp * y1 + cp * z];
}

function w3Project(p, cv) {
  const [xv, yv, zv] = w3Rot(p);
  const depth = W3.dist - zv;
  if (depth < 10) return null;
  const f = 1.1 * cv.height;
  return [cv.width / 2 + f * xv / depth,
          cv.height / 2 - f * yv / depth, depth];
}

function w3Render() {
  const data = W3.data;
  const cv = $("wcsCanvas");
  if (!data) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 900;
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(Math.min(560, cssW * 0.55) * dpr);
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1014";
  ctx.fillRect(0, 0, cv.width, cv.height);
  const M = data.marker_mm || 40;
  const line = (a, b, color, width = 1.4 * dpr) => {
    const pa = w3Project(a, cv), pb = w3Project(b, cv);
    if (!pa || !pb) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  };
  const label = (p, text, color, size = 12) => {
    const pp = w3Project(p, cv);
    if (!pp) return;
    ctx.font = `${size * dpr}px system-ui`;
    ctx.textAlign = "center";
    ctx.lineWidth = 3 * dpr;
    ctx.strokeStyle = "#000";
    ctx.strokeText(text, pp[0], pp[1]);
    ctx.fillStyle = color;
    ctx.fillText(text, pp[0], pp[1]);
  };
  // grid on the world tag's plane (z = 0)
  const ext = Math.max(4 * M, Math.ceil(W3.sceneR / (2 * M)) * 2 * M);
  const step = 2 * M;
  for (let v = -ext; v <= ext; v += step) {
    line([v, -ext, 0], [v, ext, 0], "#232a34", 1 * dpr);
    line([-ext, v, 0], [ext, v, 0], "#232a34", 1 * dpr);
  }
  // world axes at the origin
  line([0, 0, 0], [1.6 * M, 0, 0], "#4f9cf7", 2 * dpr);
  line([0, 0, 0], [0, 1.6 * M, 0], "#4bd66a", 2 * dpr);
  line([0, 0, 0], [0, 0, 1.6 * M], "#e08a3c", 2 * dpr);
  label([1.9 * M, 0, 0], "X", "#4f9cf7", 11);
  label([0, 1.9 * M, 0], "Y", "#4bd66a", 11);
  label([0, 0, 1.9 * M], "Z", "#e08a3c", 11);
  // tags — earlier snapshots' block poses draw as faint ghosts
  const lastSnap = (data.snap_count || 1) - 1;
  for (const t of data.tags) {
    const q = t.corners_world;
    const root = t.id === data.root;
    const ghost = !root && t.snap != null && t.snap !== lastSnap;
    if (ghost) {
      for (let i = 0; i < 4; i++) {
        line(q[i], q[(i + 1) % 4], "#3c4452", 1 * dpr);
      }
      continue;
    }
    line(q[1], q[2], root ? "#e7c545" : "#9aa4b2", 2 * dpr);
    line(q[2], q[3], root ? "#e7c545" : "#9aa4b2", 2 * dpr);
    line(q[0], q[1], "#4f9cf7", 2.2 * dpr);      // X edge
    line(q[0], q[3], "#4bd66a", 2.2 * dpr);      // Y edge
    const o = w3Project(q[0], cv);
    if (o) {
      ctx.fillStyle = "#ff4fd8";
      ctx.beginPath();
      ctx.arc(o[0], o[1], 3.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
    const c = [0, 1, 2].map((i) =>
      (q[0][i] + q[1][i] + q[2][i] + q[3][i]) / 4);
    label(c, String(t.id), root ? "#e7c545" : "#fff", 13);
  }
  // cameras: frustum + axis stubs + label
  for (const cam of data.cameras) {
    const T = cam.T;
    const o = [T[0][3], T[1][3], T[2][3]];
    const ax = (col, k) => [o[0] + k * T[0][col],
                            o[1] + k * T[1][col], o[2] + k * T[2][col]];
    const at = (x, y, z) => [
      o[0] + x * T[0][0] + y * T[0][1] + z * T[0][2],
      o[1] + x * T[1][0] + y * T[1][1] + z * T[1][2],
      o[2] + x * T[2][0] + y * T[2][1] + z * T[2][2]];
    const d = 1.6 * M, hw = 1.1 * M, hh = 0.75 * M;
    const rect = [at(-hw, -hh, d), at(hw, -hh, d),
                  at(hw, hh, d), at(-hw, hh, d)];
    for (let i = 0; i < 4; i++) {
      line(o, rect[i], "#7fb2f2", 1.6 * dpr);
      line(rect[i], rect[(i + 1) % 4], "#7fb2f2", 1.6 * dpr);
    }
    line(o, ax(0, 0.8 * M), "#4f9cf7", 2 * dpr);
    line(o, ax(1, 0.8 * M), "#4bd66a", 2 * dpr);
    const info = TR.cams.find((x) => x.cam.node === cam.node);
    label(at(0, -1.6 * hh, d), (info?.sel?.label ? info.sel.label + " · " : "") +
      `video${cam.node}`, "#cfe1fa", 12);
  }
}

function w3Fit(data) {
  const pts = [];
  for (const t of data.tags) pts.push(...t.corners_world);
  for (const c of data.cameras) pts.push(c.pos);
  if (!pts.length) return;
  const lo = [0, 1, 2].map((i) => Math.min(...pts.map((p) => p[i])));
  const hi = [0, 1, 2].map((i) => Math.max(...pts.map((p) => p[i])));
  W3.target = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
  const r = Math.max(200, ...[0, 1, 2].map((i) => hi[i] - lo[i]));
  W3.sceneR = r;
  W3.dist = 2.4 * r;
}

(function w3Mouse() {
  const cv = $("wcsCanvas");
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("mousedown", (e) => {
    W3.drag = { x: e.clientX, y: e.clientY,
                pan: e.button === 1 || e.button === 2 || e.ctrlKey };
    e.preventDefault();          // also suppresses middle-click autoscroll
  });
  window.addEventListener("mousemove", (e) => {
    if (!W3.drag || !W3.data) return;
    const dx = e.clientX - W3.drag.x, dy = e.clientY - W3.drag.y;
    W3.drag.x = e.clientX;
    W3.drag.y = e.clientY;
    if (W3.drag.pan) {
      // grab-the-world panning: content follows the cursor
      const k = W3.dist / (1.1 * $("wcsCanvas").height) *
                (window.devicePixelRatio || 1);
      const cy = Math.cos(W3.yaw), sy = Math.sin(W3.yaw);
      const cp = Math.cos(W3.pitch), sp = Math.sin(W3.pitch);
      W3.target[0] += -dx * k * cy + dy * k * sy * cp;
      W3.target[1] += dx * k * sy + dy * k * cy * cp;
      W3.target[2] += dy * k * sp;
    } else {
      W3.yaw -= dx * 0.008;
      W3.pitch = Math.min(Math.PI, Math.max(-Math.PI,
        W3.pitch - dy * 0.008));
    }
    w3Render();
  });
  window.addEventListener("mouseup", () => { W3.drag = null; });
  cv.addEventListener("wheel", (e) => {
    if (!W3.data) return;
    e.preventDefault();
    W3.dist *= e.deltaY > 0 ? 1.12 : 1 / 1.12;
    W3.dist = Math.min(60000, Math.max(100, W3.dist));
    w3Render();
  }, { passive: false });
})();

function wcsThumb(node, view) {
  const info = TR.cams.find((x) => x.cam.node === node);
  const rot = info?.rot || 0, swap = rot % 180 !== 0;
  const div = document.createElement("div");
  div.className = "wcs-thumb";
  const cvs = document.createElement("canvas");
  div.appendChild(cvs);
  const cap = document.createElement("div");
  cap.className = "cap dim small";
  cap.textContent = (info?.sel?.label ? info.sel.label + " · " : "") +
    `video${node} · ${view.tags.length} tag${view.tags.length === 1 ? "" : "s"}`;
  div.appendChild(cap);
  const img = new Image();
  img.onload = () => {
    const W = img.width, H = img.height;
    const draw = (scale) => {
      const c = document.createElement("canvas");
      c.width = (swap ? H : W) * scale;
      c.height = (swap ? W : H) * scale;
      const ctx = c.getContext("2d");
      ctx.save();
      ctx.scale(scale, scale);
      ctx.translate((swap ? H : W) / 2, (swap ? W : H) / 2);
      ctx.rotate(rot * Math.PI / 180);
      ctx.drawImage(img, -W / 2, -H / 2);
      ctx.restore();
      const P = (pt) => {
        const [x, y] = tkRotPt(pt[0], pt[1], rot, W, H);
        return [x * scale, y * scale];
      };
      ctx.lineWidth = Math.max(1.2, 2 * scale);
      for (const t of view.tags) {
        const q = t.corners.map(P);
        ctx.strokeStyle = "#9aa4b2";
        ctx.beginPath();
        ctx.moveTo(q[1][0], q[1][1]);
        ctx.lineTo(q[2][0], q[2][1]);
        ctx.lineTo(q[3][0], q[3][1]);
        ctx.stroke();
        ctx.strokeStyle = "#4f9cf7";
        ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]);
        ctx.lineTo(q[1][0], q[1][1]); ctx.stroke();
        ctx.strokeStyle = "#4bd66a";
        ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]);
        ctx.lineTo(q[3][0], q[3][1]); ctx.stroke();
        ctx.fillStyle = "#ff4fd8";
        ctx.beginPath();
        ctx.arc(q[0][0], q[0][1], 3.5 * Math.max(scale, 0.6), 0, Math.PI * 2);
        ctx.fill();
        const dist = t.distance_mm == null ? "" :
          " · " + (t.approx ? "~" : "") + (t.distance_mm >= 1000
            ? (t.distance_mm / 1000).toFixed(2) + " m"
            : Math.round(t.distance_mm) + " mm");
        const fs = Math.max(11, 14 * scale);
        ctx.font = `${fs}px system-ui`;
        ctx.textAlign = "left";
        const lx = Math.max(...q.map((p) => p[0])) + 4;
        const ly = Math.min(...q.map((p) => p[1])) + fs;
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#000";
        ctx.strokeText(`ID ${t.id}${dist}`, lx, ly);
        ctx.fillStyle = "#ffee33";
        ctx.fillText(`ID ${t.id}${dist}`, lx, ly);
      }
      return c;
    };
    const small = draw(280 / (swap ? H : W));
    cvs.width = small.width;
    cvs.height = small.height;
    cvs.getContext("2d").drawImage(small, 0, 0);
    div.addEventListener("click", () =>
      openLightbox(draw(1).toDataURL("image/jpeg", 0.9)));
  };
  img.src = "data:image/jpeg;base64," + view.jpg_b64;
  return div;
}

function renderWcsThumbs(data) {
  const box = $("wcsThumbs");
  box.innerHTML = "";
  const snaps = data.views_by_snap || [];
  snaps.forEach((views, si) => {
    if (snaps.length > 1) {
      const head = document.createElement("div");
      head.className = "wcs-snap-head dim small";
      head.textContent = `Snapshot ${si + 1}`;
      box.appendChild(head);
    }
    for (const [node, v] of Object.entries(views)) {
      if (v.jpg_b64) box.appendChild(wcsThumb(+node, v));
    }
  });
}

async function handleWorldData(data) {
  if (!data.ok) {
    $("wcsNote").textContent = data.error || "solve failed";
    const hasViews = (data.views_by_snap || []).some(
      (v) => Object.keys(v).length);
    renderWcsThumbs(data);
    return;
  }
  W3.data = data;
  w3Fit(data);
  $("wcsViewLabel").textContent = "world";
  w3Render();
  const omitted = [
    ...data.unlinked_cameras.map((n) => `video${n}`),
    ...data.unlinked_tags.map((t) => `tag ${t}`)];
  const uniqueTags = new Set(data.tags.map((t) => t.id)).size;
  $("wcsNote").textContent =
    `${data.cameras.length} camera(s), ${uniqueTags} tag(s) in world` +
    (data.snap_count > 1 ? ` from ${data.snap_count} snapshots` : "") +
    (data.rms_px != null ? ` · fit ${data.rms_px} px` : "") +
    (data.anchor != null && data.anchor !== data.root
      ? ` · anchored on tag ${data.anchor}` : "") +
    (omitted.length ? ` — omitted (no path to ${data.root}): ${omitted.join(", ")}` : "");
  renderWcsThumbs(data);
  const { saved, skipped } = await saveWorldPoses(data);
  $("wcsNote").textContent += saved.length
    ? ` · pose saved to ${saved.length} calibration file(s)` : "";
  if (skipped.length) {
    toast("World pose not saved for: " + skipped.join(", "), true);
  }
  tkRenderCamList();
  renderRefCamOptions();
}

/* ---- solved camera poses persist into each camera's calibration ----
   The world pose belongs with the camera it describes, so a restart (or
   another machine reading the same store) comes back with the rig already
   posed. Re-running pose estimation overwrites it; "Verify camera
   positions" checks it still holds. */
async function storePutCalibrationFor(slug, cal) {
  try {
    const r = await fetch(`api/host/calibrations/${slug}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cal) });
    return await r.json();
  } catch (e) { return { error: e.message }; }
}

function worldPoseRecord(entry, data) {
  return {
    T_world_cam: entry.T,          // 4x4 row-major, camera coords -> world
    position_mm: entry.pos,
    world_tag_id: data.root,
    anchor_tag_id: data.anchor ?? null,
    marker_mm: data.marker_mm ?? null,
    rms_px: data.rms_px ?? null,
    snapshots: data.snap_count || 1,
    tags_seen: entry.seen || [],
    solved_at: new Date().toISOString(),
    note: "T_world_cam maps camera coordinates to world coordinates " +
          "(row-major 4x4, mm). World origin is the world tag's pose. " +
          "Re-run pose estimation if this camera is moved.",
  };
}

async function saveWorldPoses(data) {
  const saved = [], skipped = [];
  for (const entry of (data.cameras || [])) {
    const c = TR.cams.find((x) => x.cam.node === entry.node);
    const slug = c?.sel?.slug;
    if (!slug) { skipped.push(`video${entry.node} (no calibration assigned)`); continue; }
    const cal = await fetchCal(slug);
    if (!cal) { skipped.push(`video${entry.node} (calibration unreadable)`); continue; }
    const rec = worldPoseRecord(entry, data);
    cal.extrinsic = { ...(cal.extrinsic || {}), world_pose: rec };
    const r = await storePutCalibrationFor(slug, cal);
    if (r && r.error) { skipped.push(`video${entry.node} (${r.error})`); continue; }
    CFG.calCache.set(slug, cal);
    if (c) c.pose = rec;
    saved.push(entry.node);
  }
  return { saved, skipped };
}

/* The world frame is anchored on a camera, not a tag: whatever the solve
   picks as its internal gauge, the result is re-expressed in the reference
   camera's frame before it ever reaches the UI. */
function worldRefBody() {
  const node = parseInt($("wcsRefCam").value, 10);
  if (Number.isNaN(node)) return null;
  return { node, mode: $("wcsRefMode").value || "topdown",
           yaw_quadrant: WCS_YAW };
}
const wcsMarker = () => parseFloat($("wcsMarkerMm").value) || 40;

/* ------------- repose the world around a top-down camera -------------
   The world tag's frame is correct but arbitrary. A camera aimed at the
   floor gives axes that mean something for a machine: Z up, Z=0 at the
   bottom of the rig. Applying it rewrites every camera's saved extrinsic,
   so live tracking and everything downstream move with it. */
let WCS_YAW = 0;

function renderRefCamOptions() {
  const sel = $("wcsRefCam");
  if (!sel) return;
  const prev = sel.value;
  const posed = TR.cams.filter((c) => c.pose?.T_world_cam);
  sel.innerHTML = posed.length
    ? posed.map((c) => {
        // how close to straight down this camera already looks, in the
        // CURRENT world — the operator shouldn't have to guess
        const T = c.pose.T_world_cam;
        const vz = -T[2][2];                       // view dir . world -Z
        const tilt = Math.acos(Math.max(-1, Math.min(1, vz))) * 180 / Math.PI;
        const name = c.sel?.label || `video${c.cam.node}`;
        return `<option value="${c.cam.node}">${esc(name)} — ${
          tilt.toFixed(0)}° off vertical</option>`;
      }).join("")
    : '<option value="">no posed cameras</option>';
  if (prev) sel.value = prev;
}

async function applyRepose() {
  const node = parseInt($("wcsRefCam").value, 10);
  if (Number.isNaN(node)) {
    toast("Pick a reference camera first.", true);
    return;
  }
  const poses = {};
  for (const c of TR.cams) {
    if (c.pose?.T_world_cam) poses[c.cam.node] = c.pose.T_world_cam;
  }
  if (Object.keys(poses).length < 1) {
    toast("No saved camera poses to repose.", true);
    return;
  }
  $("wcsNote").textContent = "reposing world…";
  let res;
  try {
    const r = await fetch("api/host/track/repose", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ poses, reference_node: node,
                             mode: $("wcsRefMode").value || "topdown",
                             yaw_quadrant: WCS_YAW }) });
    res = await r.json();
  } catch (e) {
    $("wcsNote").textContent = "repose failed: " + e.message;
    return;
  }
  if (!res.ok) {
    $("wcsNote").textContent = "repose failed: " + (res.error || "unknown");
    return;
  }
  // write the new frame into every camera's calibration
  const saved = [], skipped = [];
  for (const [n, T] of Object.entries(res.poses)) {
    const c = TR.cams.find((x) => String(x.cam.node) === String(n));
    const slug = c?.sel?.slug;
    if (!slug || !c.pose) { skipped.push(`video${n}`); continue; }
    const cal = await fetchCal(slug);
    if (!cal) { skipped.push(`video${n}`); continue; }
    const rec = { ...c.pose, T_world_cam: T,
      position_mm: [T[0][3], T[1][3], T[2][3]].map((v) => Math.round(v * 10) / 10),
      reposed: {
        reference_node: res.reference_node,
        floor_node: res.floor_node,
        yaw_quadrant: res.yaw_quadrant,
        at: new Date().toISOString(),
        note: res.note,
      } };
    cal.extrinsic = { ...(cal.extrinsic || {}), world_pose: rec };
    const w = await storePutCalibrationFor(slug, cal);
    if (w && w.error) { skipped.push(`video${n} (${w.error})`); continue; }
    CFG.calCache.set(slug, cal);
    c.pose = rec;
    saved.push(n);
  }
  const refName = TR.cams.find((c) => c.cam.node === res.reference_node)
    ?.sel?.label || `video${res.reference_node}`;
  $("wcsNote").textContent =
    `world reposed on ${refName} (was ${res.ref_tilt_from_old_down_deg}° ` +
    `off vertical) · Z up, Z=0 at video${res.floor_node} · ` +
    `yaw ${res.yaw_quadrant * 90}° · saved to ${saved.length} calibration(s)`;
  if (skipped.length) toast("Not saved for: " + skipped.join(", "), true);
  tkRenderCamList();
  renderRefCamOptions();
  // redraw the 3D view in the new frame if a solve is on screen
  if (W3.data) {
    const X = res.transform;
    const ap = (p) => [
      X[0][0] * p[0] + X[0][1] * p[1] + X[0][2] * p[2] + X[0][3],
      X[1][0] * p[0] + X[1][1] * p[1] + X[1][2] * p[2] + X[1][3],
      X[2][0] * p[0] + X[2][1] * p[1] + X[2][2] * p[2] + X[2][3]];
    for (const t of (W3.data.tags || [])) {
      t.corners_world = t.corners_world.map(ap);
    }
    for (const c of (W3.data.cameras || [])) {
      if (res.poses[c.node]) {
        c.T = res.poses[c.node];
        c.pos = [c.T[0][3], c.T[1][3], c.T[2][3]].map((v) => Math.round(v * 10) / 10);
      }
    }
    w3Fit(W3.data);
    w3Render();
  }
}

$("wcsReposeBtn").addEventListener("click", applyRepose);
$("wcsYawBtn").addEventListener("click", () => {
  WCS_YAW = (WCS_YAW + 1) % 4;
  $("wcsYawNote").textContent = `${WCS_YAW * 90}°`;
  applyRepose();
});
$("wcsRefMode").addEventListener("change", () => {
  $("wcsNote").textContent =
    "mounting changed — Apply to saved poses, or re-run World Calibration";
});

/* ---------------- verify the saved poses still hold ----------------
   Show the test block to several cameras at once: each maps it into world
   coordinates through its saved extrinsic, and they are compared against
   each other. A camera that was bumped disagrees on every tag it can see. */
const VERIFY_STATUS = {
  ok:           { icon: "✔", cls: "v-ok",    label: "verified" },
  moved:        { icon: "⚠", cls: "v-bad",   label: "MOVED" },
  disagree:     { icon: "⚠", cls: "v-bad",   label: "disagrees" },
  not_seen:     { icon: "—", cls: "v-skip",  label: "not verified" },
  unverifiable: { icon: "—", cls: "v-skip",  label: "not verified" },
  no_pose:      { icon: "◇", cls: "v-skip",  label: "no saved pose" },
};

function renderVerify(res) {
  const box = $("wcsVerifyBox");
  box.classList.remove("hidden");
  if (!res.ok) {
    box.innerHTML = `<p class="v-bad">Verification failed: ${esc(res.error || "unknown error")}</p>`;
    return;
  }
  const rows = (res.cameras || []).map((c) => {
    const s = VERIFY_STATUS[c.status] || VERIFY_STATUS.not_seen;
    const tk = TR.cams.find((x) => x.cam.node === c.node);
    const name = (tk?.sel?.label ? tk.sel.label + " — " : "") +
                 (tk?.cam.name || `node ${c.node}`);
    const err = c.max_offset_mm != null
      ? `${c.max_offset_mm} mm${c.reproj_rms_px != null
          ? ` · ${c.reproj_rms_px} px` : ""}` : "—";
    return `<tr class="${s.cls}">
      <td>${s.icon} ${esc(s.label)}</td>
      <td>${esc(name)} <span class="dim">video${c.node}</span></td>
      <td class="v-num">${esc(err)}</td>
      <td class="dim small">${esc(c.detail || "")}</td></tr>`;
  }).join("");
  const nMoved = (res.moved || []).length;
  const nOk = (res.verified || []).length;
  const unchecked = (res.cameras || [])
    .filter((c) => ["not_seen", "unverifiable", "no_pose"].includes(c.status))
    .map((c) => `video${c.node}`);
  let head;
  if (nMoved) {
    head = `<p class="v-bad"><b>⚠ ${nMoved} camera(s) appear to have moved.</b>
      Re-run pose estimation (World Calibration) before trusting 3D results.</p>`;
  } else if (nOk) {
    head = `<p class="v-ok"><b>✔ ${nOk} camera(s) verified in place.</b></p>`;
  } else {
    head = `<p class="v-skip"><b>Nothing could be verified.</b> The check needs
      one tag visible to at least two cameras that both have saved poses.</p>`;
  }
  if (unchecked.length) {
    head += `<p class="dim small">Not covered by this check: ${
      esc(unchecked.join(", "))} — these were not compared and may still
      need pose estimation.</p>`;
  }
  const ref = res.reference_group || [];
  box.innerHTML = head +
    `<table class="v-table"><tbody>${rows}</tbody></table>
     <p class="dim small">Tolerance ${res.tol_mm} mm ·
       shared tag(s): ${res.shared_tags?.length ? esc(res.shared_tags.join(", ")) : "none"} ·
       reference group: ${ref.length
         ? esc(ref.map((n) => "video" + n).join(", ")) : "none"}.
       Cameras are compared against each other, so a rigid move of the whole
       rig would not show up here.</p>`;
}

$("wcsVerifyBtn").addEventListener("click", async () => {
  const btn = $("wcsVerifyBtn");
  const poses = {};
  let nPosed = 0;
  for (const c of TR.cams) {
    if (!c.enabled) continue;
    const T = c.pose?.T_world_cam;
    if (T) { poses[c.cam.node] = T; nPosed++; }
  }
  if (nPosed < 2) {
    toast("Verification needs at least two cameras with saved poses — " +
          "run pose estimation first.", true);
    return;
  }
  btn.disabled = true;
  $("wcsNote").textContent = "verifying camera positions…";
  try {
    const r = await fetch("api/host/track/verify", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        poses,
        marker_mm: parseFloat($("tkMarkerMm").value) || 40 }) });
    const res = await r.json();
    renderVerify(res);
    $("wcsNote").textContent = res.ok
      ? `verified ${res.verified.length}, flagged ${res.moved.length}`
      : "verification failed";
    if (res.ok) {
      // show what was actually just measured, not the last solve
      W3.data = {
        marker_mm: res.marker_mm,
        tags: (res.tags || []).map((t) => ({ id: t.id, snap: 0,
                                             corners_world: t.corners_world })),
        cameras: TR.cams.filter((c) => c.pose?.T_world_cam &&
                                       res.camera_poses?.[c.cam.node])
          .map((c) => ({ node: c.cam.node, T: c.pose.T_world_cam,
                         pos: res.camera_poses[c.cam.node], seen: [] })),
        unlinked_cameras: [], unlinked_tags: [], root: null, rms_px: null,
      };
      $("wcsViewLabel").textContent =
        `verification — ${(res.tags || []).length} tag(s) as measured just now`;
      w3Fit(W3.data);
      w3Render();
      if (res.views) renderWcsThumbs({ views_by_snap: [res.views] });
    }
  } catch (e) {
    $("wcsNote").textContent = "verification failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
});

/* --- multi-snapshot world calibration: fixed cameras, moving block --- */
const WCAL = { on: false, n: 0 };

function updateWcalUI() {
  $("wcalStartBtn").classList.toggle("hidden", WCAL.on);
  for (const id of ["wcalSnapBtn", "wcalSolveBtn", "wcalCancelBtn"]) {
    $(id).classList.toggle("hidden", !WCAL.on);
  }
  $("wcalSolveBtn").textContent =
    `✔ Finish (${WCAL.n} frame${WCAL.n === 1 ? "" : "s"})`;
  $("wcalSolveBtn").disabled = WCAL.n === 0;
}

$("wcalStartBtn").addEventListener("click", async () => {
  try {
    await fetch("api/host/track/wcal/start", { method: "POST" });
  } catch { return; }
  WCAL.on = true;
  WCAL.n = 0;
  updateWcalUI();
  $("wcsNote").textContent =
    "world calibration — position the block, then Space/📸 per snapshot";
});

async function wcalSnap() {
  if (!WCAL.on) return;
  try {
    const r = await fetch("api/host/track/wcal/snap", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        marker_mm: wcsMarker() }) });
    const res = await r.json();
    if (!res.ok) throw new Error(res.error || "snapshot failed");
    WCAL.n = res.index;
    const sum = Object.entries(res.summary || {})
      .map(([n, c]) => `video${n}: ${c} tag${c === 1 ? "" : "s"}`).join(", ");
    $("wcsNote").textContent =
      `snapshot ${res.index} captured — ${sum || "⚠ no tags seen!"}`;
    updateWcalUI();
  } catch (e) {
    toast("Snapshot failed: " + e.message, true);
  }
}
$("wcalSnapBtn").addEventListener("click", wcalSnap);

$("wcalSolveBtn").addEventListener("click", async () => {
  $("wcalSolveBtn").disabled = true;
  $("wcsNote").textContent = `solving ${WCAL.n} snapshots…`;
  try {
    const r = await fetch("api/host/track/wcal/solve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: "auto", marker_mm: wcsMarker(),
        world_ref: worldRefBody() }) });
    handleWorldData(await r.json());
  } catch (e) {
    $("wcsNote").textContent = "failed: " + e.message;
  }
  WCAL.on = false;
  WCAL.n = 0;
  updateWcalUI();
});

$("wcalCancelBtn").addEventListener("click", () => {
  WCAL.on = false;
  WCAL.n = 0;
  updateWcalUI();
  $("wcsNote").textContent = "world calibration cancelled";
});

/* ------------------------------------------------------------- charuco tab */
/* One dictionary for everything: AprilTag 36h11 (587 ids). The quickstart
   board is for calibration (and later extrinsic pose); the tag sheet makes
   standalone object tags. Generation runs in Pyodide — works on Pages too. */
const PAPER_MM = { letter: [216, 279], a4: [210, 297], a3: [297, 420] };
const CH_MARGIN_MM = 10;
const CH_DPI_PRINT = 300, CH_DPI_PREVIEW = 100;
const CH_MAX_ID = 586;                   // dictionary size - 1

const CH = {
  tags: LS.get("cvcal:charuco:tags", []),    // [{id, mm}]
  boardUrl: null, busy: false, boardMaxId: 39,
  tagUrls: new Map(),                        // "id@mm" -> blob url
};

function pyBytes(proxy) {
  const v = proxy.toJs ? proxy.toJs() : proxy;
  proxy.destroy?.();
  return v instanceof Uint8Array ? v : new Uint8Array(v.buffer || v);
}

function chBoardParams() {
  const sx = Math.min(20, Math.max(3, parseInt($("chSquaresX").value, 10) || 8));
  const sy = Math.min(24, Math.max(3, parseInt($("chSquaresY").value, 10) || 10));
  const paper = $("chPaper").value in PAPER_MM ? $("chPaper").value : "letter";
  const [pw, ph] = PAPER_MM[paper];
  let square;
  if ($("chFit").checked) {
    // largest whole-mm square that keeps the board inside the margins
    square = Math.floor(Math.min((pw - 2 * CH_MARGIN_MM) / sx,
                                 (ph - 2 * CH_MARGIN_MM) / sy));
    $("chSquareMm").value = square;
  } else {
    square = Math.max(5, parseFloat($("chSquareMm").value) || 23);
  }
  const pct = Math.min(85, Math.max(50, parseFloat($("chMarkerPct").value) || 70));
  const marker = Math.round(square * pct / 100 * 2) / 2;   // 0.5 mm steps
  return { sx, sy, square, marker, paper,
           w: sx * square + 2 * CH_MARGIN_MM,
           h: sy * square + 2 * CH_MARGIN_MM };
}

function chSaveForm() {
  const p = chBoardParams();
  LS.set("cvcal:charuco:board", { paper: p.paper, sx: p.sx, sy: p.sy,
    fit: $("chFit").checked, square: $("chSquareMm").value,
    pct: $("chMarkerPct").value });
}
(function chRestoreForm() {
  const b = LS.get("cvcal:charuco:board", null);
  if (!b) return;
  if (b.paper in PAPER_MM) $("chPaper").value = b.paper;
  if (b.sx) $("chSquaresX").value = b.sx;
  if (b.sy) $("chSquaresY").value = b.sy;
  $("chFit").checked = b.fit !== false;
  if (b.square) $("chSquareMm").value = b.square;
  if (b.pct) $("chMarkerPct").value = b.pct;
})();

async function chRefreshBoard() {
  if (!S.pyReady) {
    $("chBoardMsg").textContent = "loading Python…";
    return;
  }
  if (CH.busy) return;
  CH.busy = true;
  try {
    const p = chBoardParams();
    $("chSquareMm").disabled = $("chFit").checked;
    await tick();
    const manifest = JSON.parse(
      py.charuco_board_manifest(p.sx, p.sy, p.square, p.marker));
    CH.manifest = manifest;
    CH.boardMaxId = Math.max(...manifest.marker_ids);
    const png = pyBytes(py.charuco_board_png(
      p.sx, p.sy, p.square, p.marker, CH_DPI_PREVIEW, CH_MARGIN_MM));
    if (CH.boardUrl) URL.revokeObjectURL(CH.boardUrl);
    CH.boardUrl = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    $("chBoardImg").src = CH.boardUrl;
    $("chBoardMsg").classList.add("hidden");
    $("chBoardDims").innerHTML =
      `<b>${p.sx} × ${p.sy} squares</b> · board ` +
      `${(p.sx * p.square).toFixed(0)} × ${(p.sy * p.square).toFixed(0)} mm ` +
      `on ${p.paper.toUpperCase()} · squares <b>${p.square} mm</b>, ` +
      `markers <b>${p.marker} mm</b> · uses marker IDs 0–${CH.boardMaxId}`;
    for (const id of ["chPrintBoard", "chDlBoard", "chDlManifest"]) {
      $(id).disabled = false;
    }
    chRenderTags();       // board ID range may have changed the warnings
  } catch (e) {
    $("chBoardMsg").textContent = "board generation failed: " + e.message;
    $("chBoardMsg").classList.remove("hidden");
  } finally {
    CH.busy = false;
  }
}

let chDeb = null;
["chPaper", "chSquaresX", "chSquaresY", "chSquareMm", "chFit", "chMarkerPct"]
  .forEach((id) => $(id).addEventListener("change", () => {
    chSaveForm();
    clearTimeout(chDeb);
    chDeb = setTimeout(chRefreshBoard, 250);
  }));

function chPyReady() {
  $("chAddTag").disabled = false;
  if (activeTab === "charuco") chRefreshBoard();
  chRenderTags();
}

/* --- printing (exact physical scale: mm-sized images, @page margin 0) --- */
function openPrintWindow(title, paper, bodyHtml) {
  const win = window.open("", "_blank");
  if (!win) {
    toast("Popup blocked — allow popups on this page to print.", true);
    return null;
  }
  const size = paper === "letter" ? "letter" : paper.toUpperCase();
  win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8">
<title>${esc(title)}</title>
<style>@page { size: ${size} portrait; margin: 0 }
html, body { margin: 0; padding: 0; background: #fff }</style></head>
<body>${bodyHtml}
<scr` + `ipt>onload = () => setTimeout(() => { focus(); print(); }, 250);
onafterprint = () => close();</scr` + `ipt></body></html>`);
  win.document.close();
  return win;
}

async function chBoardFullPng() {
  const p = chBoardParams();
  await tick();
  return { p, png: pyBytes(py.charuco_board_png(
    p.sx, p.sy, p.square, p.marker, CH_DPI_PRINT, CH_MARGIN_MM)) };
}

async function chPrintBoard() {
  if (!S.pyReady) return;
  const { p, png } = await chBoardFullPng();
  const url = URL.createObjectURL(new Blob([png], { type: "image/png" }));
  openPrintWindow(`ChArUco ${p.sx}x${p.sy} (${p.square}mm, 36h11)`, p.paper,
    `<img src="${url}" style="width:${p.w}mm;display:block">`);
}
$("chPrintBoard").addEventListener("click", chPrintBoard);
$("chBoardImg").addEventListener("click", chPrintBoard);

$("chDlBoard").addEventListener("click", async () => {
  if (!S.pyReady) return;
  const { p, png } = await chBoardFullPng();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([png], { type: "image/png" }));
  a.download = `charuco_${p.sx}x${p.sy}_${p.square}mm_36h11.png`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

$("chDlManifest").addEventListener("click", () => {
  if (!CH.manifest) return;
  const p = chBoardParams();
  downloadJson(CH.manifest, `charuco_${p.sx}x${p.sy}_${p.square}mm_36h11.json`);
});

/* --- individual object tags --- */
function chTagUrl(id, mm, dpi) {
  const key = `${id}@${mm}@${dpi}`;
  if (!CH.tagUrls.has(key)) {
    const png = pyBytes(py.aruco_tag_png(id, mm, dpi));
    CH.tagUrls.set(key,
      URL.createObjectURL(new Blob([png], { type: "image/png" })));
  }
  return CH.tagUrls.get(key);
}

function chRenderTags() {
  const list = $("chTagList");
  list.innerHTML = "";
  for (const t of CH.tags) {
    const div = document.createElement("div");
    div.className = "tag-tile";
    const clash = t.id <= CH.boardMaxId;
    div.innerHTML = `<img alt="tag ${t.id}">
      <div class="cap">ID ${t.id} · ${t.mm} mm</div>
      ${clash ? '<div class="clash">⚠ board ID range</div>' : ""}
      <button class="x" title="Remove from sheet">✕</button>`;
    if (S.pyReady) {
      // preview at ~150 px regardless of physical size
      const dpi = Math.max(40, Math.round(150 * 25.4 / t.mm));
      try { div.querySelector("img").src = chTagUrl(t.id, t.mm, dpi); }
      catch { /* bad id — leave blank */ }
    }
    div.querySelector(".x").addEventListener("click", () => {
      CH.tags = CH.tags.filter((x) => x.id !== t.id);
      LS.set("cvcal:charuco:tags", CH.tags);
      chRenderTags();
    });
    list.appendChild(div);
  }
  $("chPrintTags").disabled = !CH.tags.length || !S.pyReady;
  $("chClearTags").disabled = !CH.tags.length;
}

$("chAddTag").addEventListener("click", () => {
  const id = parseInt($("chTagId").value, 10);
  const mm = Math.min(200, Math.max(10, parseFloat($("chTagMm").value) || 40));
  if (!(id >= 0 && id <= CH_MAX_ID)) {
    toast(`Tag ID must be 0–${CH_MAX_ID} (36h11 dictionary).`, true);
    return;
  }
  if (CH.tags.some((t) => t.id === id)) {
    toast(`Tag ${id} is already on the sheet.`, true);
    return;
  }
  CH.tags.push({ id, mm });
  LS.set("cvcal:charuco:tags", CH.tags);
  let next = id + 1;                       // advance to the next free ID
  while (next <= CH_MAX_ID && CH.tags.some((t) => t.id === next)) next++;
  $("chTagId").value = Math.min(next, CH_MAX_ID);
  chRenderTags();
});

$("chClearTags").addEventListener("click", () => {
  if (!confirm(`Remove all ${CH.tags.length} tags from the sheet?`)) return;
  CH.tags = [];
  LS.set("cvcal:charuco:tags", []);
  chRenderTags();
});

$("chPrintTags").addEventListener("click", async () => {
  if (!S.pyReady || !CH.tags.length) return;
  await tick();
  const cells = CH.tags.map((t) => {
    const url = chTagUrl(t.id, t.mm, CH_DPI_PRINT);
    // 5 mm white quiet zone inside the dashed cut line
    return `<div style="padding:5mm;border:0.3mm dashed #888;margin:2mm;
        text-align:center;break-inside:avoid">
      <img src="${url}" style="width:${t.mm}mm;display:block;margin:0 auto">
      <div style="font:7pt sans-serif;margin-top:0.8mm">ID ${t.id} · ${t.mm}mm</div>
    </div>`;
  }).join("");
  openPrintWindow(`ArUco tags ×${CH.tags.length} (36h11)`,
    $("chPaper").value,
    `<div style="display:flex;flex-wrap:wrap;align-items:flex-start;
        align-content:flex-start;padding:8mm">${cells}</div>`);
});

/* ---------------------------------------------------------------- lightbox */
function openLightbox(src) {
  $("lightboxImg").src = src;
  $("lightbox").classList.remove("hidden");
}
function closeLightbox() { $("lightbox").classList.add("hidden"); }
$("lightbox").addEventListener("click", (e) => {
  if (e.target.id !== "lightboxImg") closeLightbox();
});
$("lightboxClose").addEventListener("click", closeLightbox);

/* ---------------------------------------------------------------- keyboard */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
  if (e.code !== "Space") return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (activeTab === "collect") {
    e.preventDefault();
    if (S.collecting) stopCollecting();
    else if (streamActive() && S.pyReady) snapCalibImage(true);
  } else if (activeTab === "measure") {
    e.preventDefault();
    // Space snaps the undistorted view (falls back to raw if no calibration)
    if (streamActive() && S.pyReady) takeSnap(S.activeCal ? "undistorted" : "raw");
  } else if (activeTab === "tracking" && WCAL.on) {
    e.preventDefault();
    wcalSnap();          // world-calibration snapshot
  }
});

/* ==================================================================== *
 *  LIVE TRACKING TAB                                                    *
 *  Detections fused into world space by the bridge and drawn live. The  *
 *  camera list mirrors the pose tab but gates on a saved world pose —   *
 *  without one a camera contributes nothing to a 3D fix, so starting    *
 *  with it checked is refused rather than silently degraded.            *
 * ==================================================================== */
const LV = {
  on: false, cams: [], dets: [], enabled: new Set(), poll: null,
  abort: null, bytes: 0, rateB: 0, rateT: 0, snap: null, starting: false,
  view: { yaw: 0.7, pitch: 0.9, dist: 2000, target: [0, 0, 0], drag: null },
};

async function lvBuildCamList() {
  // reuse the pose tab's discovery; it already resolves calibration + pose
  if (!TR.cams.length) await tkBuildCams();
  LV.cams = TR.cams.map((c) => ({
    cam: c.cam, sel: c.sel, res: c.res, pose: c.pose, calOk: c.calOk,
    rot: c.rot || 0,          // same view rotation the other tabs apply
    canvas: null, stat: null, busy: false,
  }));
  for (const c of LV.cams) {
    const key = `cvcal:liveoff:${tkCamKey(c.cam)}`;
    if (!rigGet(key, false) && c.pose) LV.enabled.add(c.cam.node);
  }
}

/* The label is the camera's identity here — four OV9782s report the same
   model string and their /dev/video numbers move between reboots, so the
   physical label is the only thing that means anything to a person. */
const lvCamLabel = (c) => c.sel?.label || null;
const lvCamSub = (c) =>
  `${c.cam.name}${c.cam.usb?.bus_path ? ` · ${c.cam.usb.bus_path}` : ""}` +
  ` · video${c.cam.node}`;

function lvCamTitle(c) {
  const l = lvCamLabel(c);
  return l ? `<b class="lv-label">${esc(l)}</b>`
           : `<b class="lv-label none">unlabeled</b>`;
}

function lvRenderCamList() {
  const box = $("lvCamList");
  box.innerHTML = "";
  if (!LV.cams.length) { box.textContent = "No cameras detected."; return; }
  for (const c of LV.cams) {
    const on = LV.enabled.has(c.cam.node);
    const row = document.createElement("div");
    row.className = "tk-camrow";
    row.innerHTML = `
      <label class="tk-camlabel"><input type="checkbox"${on ? " checked" : ""}>
        <span>${lvCamTitle(c)}
          <span class="dim small">${esc(lvCamSub(c))}</span></span></label>
      <span class="tk-pose ${c.pose ? "has" : "none"}">${
        c.pose ? "◈ posed" : "◇ no pose"}</span>`;
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) LV.enabled.add(c.cam.node);
      else LV.enabled.delete(c.cam.node);
      rigSet(`cvcal:liveoff:${tkCamKey(c.cam)}`, !e.target.checked);
      lvRenderThumbs();
    });
    box.appendChild(row);
  }
}

async function lvLoadDetectors() {
  let data = { detectors: [] };
  try { data = await (await fetch("api/host/detectors")).json(); } catch {}
  LV.dets = data.detectors || [];
  const box = $("lvDetList");
  box.innerHTML = LV.dets.map((d) => {
    const dis = d.available ? "" : " disabled";
    const checked = d.available && d.key === "aruco" ? " checked" : "";
    return `<label class="lv-det${d.available ? "" : " off"}">
      <input type="checkbox" value="${esc(d.key)}"${checked}${dis}>
      <span><b>${esc(d.name)}</b>${d.requires_gpu
        ? ' <span class="lv-gpu">GPU</span>' : ""}
        <span class="dim small">${esc(d.description)}</span>
        ${d.reason ? `<span class="dim small">${esc(d.reason)}${
          d.install_hint && !d.available
            ? ` — <code>${esc(d.install_hint)}</code>` : ""}</span>` : ""}
      </span></label>`;
  }).join("") || '<span class="dim">No detectors reported.</span>';
}

const lvSelectedDetectors = () =>
  [...$("lvDetList").querySelectorAll("input:checked")].map((i) => i.value);

function lvRenderThumbs() {
  const box = $("lvThumbs");
  box.innerHTML = "";
  for (const c of LV.cams) {
    if (!LV.enabled.has(c.cam.node)) { c.canvas = c.stat = null; continue; }
    const card = document.createElement("div");
    card.className = "lv-card";
    card.innerHTML = `
      <div class="tk-cap small">${lvCamTitle(c)}
        <span class="dim">${esc(lvCamSub(c))}</span></div>
      <canvas width="16" height="9"></canvas>
      <div class="tk-stat dim small">idle</div>`;
    box.appendChild(card);
    c.canvas = card.querySelector("canvas");
    c.stat = card.querySelector(".tk-stat");
  }
  if (!box.children.length) {
    box.innerHTML = '<p class="dim">No cameras selected.</p>';
  }
}

/* ------------------------------------------------- live 3D view (own state) */
function lv3Rot(p) {
  const V = LV.view;
  const x = p[0] - V.target[0], y = p[1] - V.target[1], z = p[2] - V.target[2];
  const cy = Math.cos(V.yaw), sy = Math.sin(V.yaw);
  const x1 = cy * x + sy * y, y1 = -sy * x + cy * y;
  const cp = Math.cos(V.pitch), sp = Math.sin(V.pitch);
  return [x1, cp * y1 + sp * z, -sp * y1 + cp * z];
}
function lv3Project(p, cv) {
  const [xv, yv, zv] = lv3Rot(p);
  const depth = LV.view.dist - zv;
  if (depth < 10) return null;
  const f = 1.1 * cv.height;
  return [cv.width / 2 + f * xv / depth, cv.height / 2 - f * yv / depth];
}

function lv3Render() {
  const cv = $("lvCanvas");
  if (!cv || !cv.clientWidth) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(cv.clientWidth * dpr);
  cv.height = Math.round(Math.min(520, cv.clientWidth * 0.52) * dpr);
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1014";
  ctx.fillRect(0, 0, cv.width, cv.height);
  const line = (a, b, color, w = 1.4 * dpr, dash = null) => {
    const pa = lv3Project(a, cv), pb = lv3Project(b, cv);
    if (!pa || !pb) return;
    ctx.setLineDash(dash || []);
    ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    ctx.setLineDash([]);
  };
  const dot = (p, color, r = 2.6 * dpr) => {
    const pp = lv3Project(p, cv);
    if (!pp) return;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(pp[0], pp[1], r, 0, 7); ctx.fill();
  };
  const text = (p, s, color, size = 12) => {
    const pp = lv3Project(p, cv);
    if (!pp) return;
    ctx.font = `${size * dpr}px system-ui`;
    ctx.textAlign = "center";
    ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "#000c";
    ctx.strokeText(s, pp[0], pp[1]); ctx.fillStyle = color;
    ctx.fillText(s, pp[0], pp[1]);
  };
  /* Ground grid on the world XY plane (z = 0, the world tag's plane).
     Without a plane to sit on, an orbiting camera view gives the eye
     nothing to judge angle or scale against — two axis stubs are not
     enough to tell "above, looking down" from "below, looking up". */
  const step = lvNiceStep(LV.view.dist / 8);
  const N = 8;
  const gx = Math.round(LV.view.target[0] / step) * step;
  const gy = Math.round(LV.view.target[1] / step) * step;
  const lo = -N * step, hi = N * step;
  for (let i = -N; i <= N; i++) {
    const o = i * step;
    const onAxisX = Math.abs(gy + o) < 1e-6;
    const onAxisY = Math.abs(gx + o) < 1e-6;
    line([gx + lo, gy + o, 0], [gx + hi, gy + o, 0],
         onAxisX ? "#3a4654" : "#232a33", (onAxisX ? 1.3 : 1) * dpr);
    line([gx + o, gy + lo, 0], [gx + o, gy + hi, 0],
         onAxisY ? "#3a4654" : "#232a33", (onAxisY ? 1.3 : 1) * dpr);
  }
  const box = lv3DrawBox(ctx, cv, dpr);
  // cameras
  for (const c of LV.cams) {
    if (!c.pose?.T_world_cam || !LV.enabled.has(c.cam.node)) continue;
    const T = c.pose.T_world_cam;
    const o = [T[0][3], T[1][3], T[2][3]];
    const ax = (k, s) => [o[0] + T[0][k] * s, o[1] + T[1][k] * s, o[2] + T[2][k] * s];
    const f = 70, corners = [];
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const d = ax(2, f);
      const rx = [T[0][0] * sx * f * 0.6, T[1][0] * sx * f * 0.6, T[2][0] * sx * f * 0.6];
      const ry = [T[0][1] * sy * f * 0.45, T[1][1] * sy * f * 0.45, T[2][1] * sy * f * 0.45];
      corners.push([d[0] + rx[0] + ry[0], d[1] + rx[1] + ry[1], d[2] + rx[2] + ry[2]]);
    }
    for (let i = 0; i < 4; i++) {
      line(o, corners[i], "#7a8896");
      line(corners[i], corners[(i + 1) % 4], "#7a8896");
    }
    text([o[0], o[1], o[2] + step * 0.3],
         lvCamLabel(c) || `video${c.cam.node}`, "#9fb0c0", 11);
  }
  // tracked items
  const byKind = { aruco: "#e7c545", hands: "#57d1c9" };
  for (const it of (LV.snap?.items || [])) {
    const pts = it.points_world;
    if (!pts) continue;
    const col = byKind[it.kind] || "#e7c545";
    const dash = it.single_view ? [5 * dpr, 4 * dpr] : null;
    const meta = LV.dets.find((d) => d.key === it.kind);
    for (const [a, b] of (meta?.edges || [])) {
      if (pts[a] && pts[b]) line(pts[a], pts[b], col, 1.6 * dpr, dash);
    }
    for (const p of pts) if (p) dot(p, col);
    // Anchor the label on the wrist for anything that has one: at the
    // centroid it sits in the middle of the fingers, which is the part
    // worth looking at. Hangs below so it never covers the hand either.
    const wi = (it.names || []).indexOf("wrist");
    const anchor = (wi >= 0 && pts[wi]) ? pts[wi] : it.center;
    if (anchor) {
      const drop = (wi >= 0 && pts[wi]) ? -step * 0.28 : step * 0.2;
      text([anchor[0], anchor[1], anchor[2] + drop], it.label, col, 11);
    }
  }
  lv3Hud(ctx, cv, dpr, step, box);
}

/* A five-sided box around the monitored volume. The cameras sit near its
   edges, so their bounding box plus a small margin stands in for it. The
   face nearest the eye is left off, so you always look INTO the box
   rather than through a translucent lid — and because it is chosen per
   frame from the current view direction, it stays open as you orbit.

   Faces are painted far-to-near: a 2D canvas has no depth buffer, so
   overlap has to be resolved by drawing order. */
function lv3DrawBox(ctx, cv, dpr) {
  const pts = [];
  for (const c of LV.cams) {
    if (!c.pose?.T_world_cam || !LV.enabled.has(c.cam.node)) continue;
    const T = c.pose.T_world_cam;
    pts.push([T[0][3], T[1][3], T[2][3]]);
  }
  if (pts.length < 2) return null;
  const M = 10;                       // a little breathing room, mm
  const lo = [0, 1, 2].map((i) => Math.min(...pts.map((p) => p[i])) - M);
  const hi = [0, 1, 2].map((i) => Math.max(...pts.map((p) => p[i])) + M);
  const corner = (i) => [(i & 4) ? hi[0] : lo[0],
                         (i & 2) ? hi[1] : lo[1],
                         (i & 1) ? hi[2] : lo[2]];
  // corner index bits are (x, y, z); each quad walks one face in order
  const FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                 [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
  const faces = FACES.map((f) => {
    const c3 = f.map(corner);
    const mid = [0, 1, 2].map((k) => c3.reduce((a, q) => a + q[k], 0) / 4);
    return { c3, depth: LV.view.dist - lv3Rot(mid)[2] };
  });
  let near = 0;
  faces.forEach((f, i) => { if (f.depth < faces[near].depth) near = i; });
  for (const f of faces.filter((_, i) => i !== near)
                       .sort((a, b) => b.depth - a.depth)) {
    const p = f.c3.map((q) => lv3Project(q, cv));
    if (p.some((q) => !q)) continue;
    ctx.beginPath();
    ctx.moveTo(p[0][0], p[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]);
    ctx.closePath();
    ctx.fillStyle = "#4f9cf714";
    ctx.fill();
    ctx.strokeStyle = "#4f9cf755";
    ctx.lineWidth = 1.2 * dpr;
    ctx.stroke();
  }
  return { lo, hi };
}

/* A "nice" 1/2/5 x 10^n step, so grid squares are a round number of mm */
function lvNiceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(x, 1e-6))));
  const m = x / p;
  return p * (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10);
}

/* Screen-space readout: which way is up, how big a grid square is, and
   where the eye is. Orbit views are ambiguous without it — the same
   picture can be "looking down from above" or "up from below". */
function lv3Hud(ctx, cv, dpr, step, box) {
  const V = LV.view;
  const pitchDeg = V.pitch * 180 / Math.PI;
  const yawDeg = ((V.yaw * 180 / Math.PI) % 360 + 360) % 360;
  const from = pitchDeg > 8 ? "from above" : pitchDeg < -8 ? "from below"
                                           : "edge-on";
  const grid = step >= 1000 ? `${(step / 1000).toFixed(step % 1000 ? 2 : 0)} m`
                            : `${step.toFixed(0)} mm`;
  const lines = [
    `grid ${grid} · view ${(V.dist / 1000).toFixed(2)} m out`,
    `looking ${from} · yaw ${yawDeg.toFixed(0)}° · pitch ${pitchDeg.toFixed(0)}°`,
  ];
  if (box) {
    const d = [0, 1, 2].map((i) => Math.round(box.hi[i] - box.lo[i]));
    lines.push(`box ${d[0]} × ${d[1]} × ${d[2]} mm`);
  }
  ctx.save();
  ctx.font = `${11 * dpr}px system-ui`;
  ctx.textAlign = "left";
  const pad = 7 * dpr;
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2;
  const h = lines.length * 15 * dpr + pad * 1.4;
  ctx.fillStyle = "#0d1014cc";
  ctx.strokeStyle = "#2a323c";
  ctx.lineWidth = dpr;
  ctx.beginPath();
  ctx.rect(pad, pad, w, h);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#9fb0c0";
  lines.forEach((l, i) =>
    ctx.fillText(l, pad * 2, pad + 15 * dpr * (i + 1) - 3 * dpr));
  ctx.restore();
}

/* Frame everything that exists: cameras, tracked items, the origin. A
   fixed default distance is meaningless when rigs differ in scale. */
function lv3Fit() {
  const pts = [[0, 0, 0]];
  for (const c of LV.cams) {
    if (!c.pose?.T_world_cam || !LV.enabled.has(c.cam.node)) continue;
    const T = c.pose.T_world_cam;
    pts.push([T[0][3], T[1][3], T[2][3]]);
  }
  for (const it of (LV.snap?.items || [])) {
    if (it.center) pts.push(it.center);
  }
  const ax = [0, 1, 2].map((i) => pts.map((p) => p[i]));
  const min = ax.map((v) => Math.min(...v));
  const max = ax.map((v) => Math.max(...v));
  LV.view.target = [0, 1, 2].map((i) => (min[i] + max[i]) / 2);
  LV.view.dist = Math.max(
    400, Math.max(...[0, 1, 2].map((i) => max[i] - min[i])) * 2.2);
}

function lvBindView() {
  const cv = $("lvCanvas");
  if (!cv || cv._bound) return;
  cv._bound = true;
  cv.addEventListener("mousedown", (e) => {
    LV.view.drag = { x: e.clientX, y: e.clientY,
                     pan: e.button !== 0 || e.ctrlKey };
    e.preventDefault();
  });
  addEventListener("mouseup", () => { LV.view.drag = null; });
  addEventListener("mousemove", (e) => {
    const d = LV.view.drag;
    if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    d.x = e.clientX; d.y = e.clientY;
    const V = LV.view;
    // Same convention as the WCS viewer above (w3 drag handler): both use
    // the same projection, so they must use the same signs — dragging is
    // muscle memory and a viewer that spins the other way reads as broken.
    if (d.pan) {
      const k = V.dist / (1.1 * $("lvCanvas").height) *
                (window.devicePixelRatio || 1);
      const cy = Math.cos(V.yaw), sy = Math.sin(V.yaw);
      const cp = Math.cos(V.pitch), sp = Math.sin(V.pitch);
      V.target[0] += -dx * k * cy + dy * k * sy * cp;
      V.target[1] += dx * k * sy + dy * k * cy * cp;
      V.target[2] += dy * k * sp;
    } else {
      V.yaw -= dx * 0.008;
      // clamped short of vertical: going over the top is what made this
      // view hard to read in the first place
      V.pitch = Math.max(-1.5, Math.min(1.5, V.pitch - dy * 0.008));
    }
    lv3Render();
  });
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    LV.view.dist = Math.max(150, Math.min(20000,
      LV.view.dist * (e.deltaY > 0 ? 1.12 : 0.89)));
    lv3Render();
  }, { passive: false });
}

/* ------------------------------------------------------------ start / stop */
$("lvStartBtn").addEventListener("click", () => (LV.on ? lvStop() : lvStart()));

async function lvStart() {
  if (LV.starting) return;
  const chosen = LV.cams.filter((c) => LV.enabled.has(c.cam.node));
  if (!chosen.length) { toast("Select at least one camera.", true); return; }
  const dets = lvSelectedDetectors();
  if (!dets.length) { toast("Select at least one thing to track.", true); return; }
  const missing = chosen.filter((c) => !c.pose?.T_world_cam);
  if (missing.length) {
    const names = missing.map((c) => `video${c.cam.node}`).join(", ");
    $("lvNote").innerHTML = `<span class="v-bad">⚠ No saved world pose for
      ${esc(names)} — uncheck them, or run Camera Pose Estimation first.</span>`;
    toast(`Cannot start: ${names} have no world pose.`, true);
    return;
  }
  // The marker size used for World Calibration sets the scale of the
  // entire reconstruction -- it is the only metric input the solve has.
  // Calibrate with the wrong size and every camera position is scaled by
  // (assumed / true). Triangulated tags stay self-consistent in that
  // scaled world, but single-view PnP here uses the size below to produce
  // a TRUE distance, which then disagrees with the scaled camera
  // positions. The result is exactly a handful of tags sitting at the
  // wrong distance while the rest cluster correctly.
  const mm = parseFloat($("lvMarkerMm").value) || 40;
  const calSizes = [...new Set(chosen
    .map((c) => c.pose?.marker_mm).filter((v) => v))];
  const mismatch = calSizes.filter((v) => Math.abs(v - mm) > 0.51);
  if (mismatch.length) {
    $("lvNote").innerHTML = `<span class="v-warn">⚠ Marker size here is
      ${mm} mm, but the world was calibrated at ${esc(mismatch.join(", "))} mm.
      That scales every camera position by ${(mismatch[0] / mm).toFixed(3)}×,
      so single-view tags will land at the wrong distance. Re-run World
      Calibration at the correct size — changing it here does not undo
      it.</span>`;
    toast(`World was calibrated at ${mismatch[0]} mm, not ${mm} mm — ` +
          "re-run World Calibration.", true);
  }
  LV.starting = true;
  if (!mismatch.length) $("lvNote").textContent = "starting…";
  try {
    const body = {
      view_fps: parseFloat($("lvViewFps").value) || 10,
      track_fps: parseFloat($("lvTrackFps").value) || 10,
      marker_mm: mm,
      detectors: dets,
      cameras: await Promise.all(chosen.map(async (c) => {
        const cal = c.sel ? await fetchCal(c.sel.slug) : null;
        const i = cal?.intrinsic || {};
        return { node: c.cam.node, width: c.res[0], height: c.res[1],
                 K: i.camera_matrix || null, dist: i.dist_coeffs || null,
                 cal_size: i.image_size || null,
                 T_world_cam: c.pose.T_world_cam };
      })),
    };
    const r = await fetch("api/host/live/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const res = await r.json();
    if (!res.ok) {
      $("lvNote").innerHTML = `<span class="v-bad">${esc(res.error || "start failed")}</span>`;
      toast(res.error || "Live tracking failed to start.", true);
      return;
    }
    LV.on = true;
    $("lvStartBtn").textContent = "⏹ Stop live tracking";
    $("lvStartBtn").classList.add("stop");
    const skipped = (res.skipped || [])
      .map((s) => `${s.key}: ${s.reason}`).join("; ");
    $("lvNote").textContent =
      `tracking ${res.detectors.join(", ")} on ${res.started.length} camera(s)` +
      (res.failed?.length ? ` — failed to open: ${res.failed.join(", ")}` : "") +
      (skipped ? ` — skipped ${skipped}` : "");
    lvBindView();
    lvOpenStream(res.started);
    LV.poll = setInterval(lvPoll, 1000 / Math.max(
      1, parseFloat($("lvTrackFps").value) || 10));
  } catch (e) {
    $("lvNote").textContent = "start failed: " + e.message;
  } finally {
    LV.starting = false;
  }
}

async function lvStop() {
  LV.on = false;
  if (LV.poll) { clearInterval(LV.poll); LV.poll = null; }
  if (LV.abort) { LV.abort.abort(); LV.abort = null; }
  $("lvStartBtn").textContent = "▶ Start live tracking";
  $("lvStartBtn").classList.remove("stop");
  $("lvNote").textContent = "stopped";
  try { await fetch("api/host/live/stop", { method: "POST" }); } catch {}
}

async function lvOpenStream(nodes) {
  const ctrl = new AbortController();
  LV.abort = ctrl;
  const fps = parseFloat($("lvViewFps").value) || 10;
  while (LV.abort === ctrl && LV.on) {
    try {
      await readFrameStream(
        `api/host/multistream?nodes=${nodes.join(",")}&width=480&quality=75` +
        `&fps=${fps}&t=${Date.now()}`, ctrl, lvFrame);
    } catch { /* aborted or bridge away */ }
    if (LV.abort !== ctrl || !LV.on) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function lvFrame(node, jpg) {
  LV.bytes += jpg.length;
  const c = LV.cams.find((x) => x.cam.node === node &&
                                LV.enabled.has(x.cam.node));
  if (!c || !c.canvas || c.busy) return;
  c.busy = true;
  try {
    const bmp = await createImageBitmap(new Blob([jpg], { type: "image/jpeg" }));
    // same rotation the collect/pose views apply, so a camera mounted
    // sideways reads the same way on every tab
    const rot = c.rot || 0, swap = rot % 180 !== 0;
    const W = bmp.width, H = bmp.height;
    const cw = swap ? H : W, ch = swap ? W : H;
    const cv = c.canvas;
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    const ctx = cv.getContext("2d");
    ctx.save();
    ctx.translate(cw / 2, ch / 2);
    ctx.rotate(rot * Math.PI / 180);
    ctx.drawImage(bmp, -W / 2, -H / 2);
    ctx.restore();
    bmp.close?.();
  } catch { /* torn frame */ } finally { c.busy = false; }
}

async function lvPoll() {
  if (!LV.on) return;
  let snap;
  try { snap = await (await fetch("api/host/live/results")).json(); }
  catch { return; }
  LV.snap = snap;
  // The 3D view is the thing worth watching at tracker rate. Numbers
  // changing ten times a second are unreadable and make the panel look
  // frantic, so text refreshes on a slow clock of its own.
  const nowMs = Date.now();
  const slow = !LV.slowAt || nowMs - LV.slowAt >= 2000;
  if (slow) LV.slowAt = nowMs;
  if (!LV.fitted && (snap.items || []).some((i) => i.center)) {
    LV.fitted = true;            // frame the rig once real geometry lands
    lv3Fit();
  }
  lv3Render();
  if (!slow) return;
  lvRenderItems(snap);
  lvRenderMetrics(snap);
  for (const c of LV.cams) {
    if (!c.stat) continue;
    const pc = (snap.per_cam || {})[c.cam.node] || {};
    const cap = (snap.capture || {})[c.cam.node] || {};
    c.stat.textContent = pc.error
      ? `⚠ ${pc.error}`
      : `${cap.fps ?? "?"} fps · ${pc.n ?? 0} det` +
        (pc.detect_ms ? " · " + Object.entries(pc.detect_ms)
          .map(([k, v]) => `${k} ${v}ms`).join(" ") : "");
    c.stat.classList.toggle("stat-error", !!pc.error);
  }
}

function lvRenderItems(snap) {
  const box = $("lvItems");
  const items = snap.items || [];
  if (!items.length) {
    box.innerHTML = '<p class="dim small">Nothing detected yet.</p>';
    return;
  }
  box.innerHTML = `<table class="v-table"><tbody>${items.map((it) => {
    const pos = it.center
      ? it.center.map((v) => v.toFixed(0)).join(", ") : "—";
    const how = it.localized === false
      ? `<span class="v-skip">${esc(it.reason || "not localized")}</span>`
      : (it.single_view ? "single view (pose)" : `${it.n_views} views`);
    return `<tr><td>${esc(it.label)}</td>
      <td class="dim small">${esc(it.kind)}</td>
      <td class="v-num">${esc(pos)}</td>
      <td class="dim small">${how}${it.rms_px != null
        ? ` · ${it.rms_px} px` : ""}</td>
      <td class="dim small">${(it.cameras || []).map((n) => "v" + n).join(" ")}</td>
      </tr>`;
  }).join("")}</tbody></table>`;
}

/* Fixed rows in fixed sections. Values appear and disappear constantly —
   a panel that adds and removes rows to match jumps around while you are
   reading it, so every row is always present and shows "—" when it has
   nothing to say. */
const LV_DASH = "—";
const lvNum = (v, f) => (v == null || Number.isNaN(v)) ? LV_DASH : f(v);

function lvRenderMetrics(snap) {
  const s = snap.stats || {}, m = snap.metrics || {}, g = snap.gpu || {};
  const sec = [];

  const perf = [];
  perf.push(["Tracker rate", lvNum(s.achieved_fps, (v) =>
    `${v} / ${s.target_fps ?? "?"} fps (${Math.min(999,
      Math.round(100 * v / (s.target_fps || 1)))}%)`)]);
  perf.push(["Tracker duty", lvNum(s.duty_pct, (v) => {
    // colour alone carries the warning: appending text here made the row
    // wrap onto a second line and shunted everything below it
    const cls = v >= 90 ? "v-bad" : v >= 70 ? "v-warn" : "";
    return `<span class="${cls}">${v}%</span>`;
  })]);
  perf.push(["Detection / tick", lvNum(s.detect_wall_ms, (v) =>
    `${v} ms · ${s.workers ?? "?"}w`)]);
  // one row per known detector, running or not, so the panel keeps its shape
  for (const d of (LV.dets || [])) {
    perf.push([`${d.key} time`,
      lvNum((s.detectors || {})[d.key], (v) => `${v} ms/camera`)]);
  }
  perf.push(["Items", s.items == null ? LV_DASH
    : `${s.localized ?? 0} localized of ${s.items}`]);
  // skew in ms IS the mm of error per m/s of motion (1 m/s = 1 mm/ms), so
  // state it that way instead of making the reader do the conversion
  perf.push(["Camera sync", lvNum(s.frame_skew_max_ms, (v) =>
    `±${Math.round(v)} mm per m/s of motion`)]);
  sec.push(["Performance", perf]);

  const sys = [];
  sys.push(["CPU", lvNum(m.proc_cpu_pct_one_core, (v) =>
    `${(v / 100).toFixed(2)} of ${m.ncpu} cores (${m.proc_cpu_pct_machine}%)`)]);
  sys.push(["System CPU", lvNum(m.system_cpu_pct, (v) => `${v}%`)]);
  sys.push(["Memory", lvNum(m.rss_mb, (v) =>
    `${v} MB${m.rss_pct != null ? ` (${m.rss_pct}%)` : ""}`)]);
  sys.push(["Load avg", m.loadavg ? m.loadavg.join(" / ") : LV_DASH]);
  let rate = null;
  const now = Date.now();
  if (LV.rateT) {
    const dt = (now - LV.rateT) / 1000;
    if (dt > 0.5) {
      LV.lastRate = (LV.bytes - LV.rateB) / dt / 125000;
      LV.rateT = now; LV.rateB = LV.bytes;
    }
    rate = LV.lastRate;
  } else { LV.rateT = now; LV.rateB = LV.bytes; }
  sys.push(["Video in", lvNum(rate, (v) => `${v.toFixed(1)} Mbit/s`)]);
  sec.push(["System", sys]);

  const gpu = (g.gpus || [])[0] || {};
  const gr = [];
  gr.push(["Device", g.available ? esc(gpu.name || "?")
    : `<span class="dim">unavailable</span>`]);
  gr.push(["Utilization", lvNum(gpu.util_pct, (v) => `${v}%`)]);
  gr.push(["Memory", lvNum(gpu.mem_used_mb, (v) =>
    `${v} / ${gpu.mem_total_mb} MB (${gpu.mem_pct}%)`)]);
  gr.push(["Temperature", lvNum(gpu.temp_c, (v) => `${v} °C`)]);
  gr.push(["Power", lvNum(gpu.power_w, (v) => `${v} W`)]);
  sec.push(["GPU", gr]);

  $("lvMetrics").innerHTML = sec.map(([title, rows]) =>
    `<div class="tk-msec">${esc(title)}</div>` + rows.map(([k, v]) =>
      `<div class="tk-mrow"><span>${esc(k)}</span><b>${v}</b></div>`).join("")
  ).join("");
}

async function lvEnter() {
  await lvBuildCamList();
  lvRenderCamList();
  await lvLoadDetectors();
  lvRenderThumbs();
  lvBindView();
  lv3Fit();                      // start framed on the posed cameras
  lv3Render();
}

$("lvResetView").addEventListener("click", () => {
  LV.view.yaw = 0.7;
  LV.view.pitch = 0.9;
  lv3Fit();
  lv3Render();
});

/* -------------------------------------------------------------------- init */
restoreForm();
(async () => {
  await detectHost();
  if (HOST) {
    document.title += " — local (host cameras)";
    $("configTabBtn").classList.remove("hidden");
    $("trackTabBtn").classList.remove("hidden");
    $("liveTabBtn").classList.remove("hidden");
    await rigLoad();                    // before any camera assignment is read
    await refreshDevices();
    switchTab("cameras");               // host-mode home page
    setInterval(pollCameras, 3000);     // notice plug/unplug in the background
    // MJPEG <img> readiness lags the stream start; nudge the UI as it lands
    setInterval(() => {
      if (sourceReady($("liveImg"))) {
        $("liveOverlayMsg").classList.add("hidden");
        updateButtons();
      }
    }, 700);
  } else if (!navigator.mediaDevices?.getUserMedia) {
    toast("This browser does not support camera access (getUserMedia).", true);
  } else {
    refreshDevices();
    navigator.mediaDevices.addEventListener?.("devicechange", refreshDevices);
  }
})();
bootPython();
