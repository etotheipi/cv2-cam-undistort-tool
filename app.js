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
  if (S.hostCam) LS.set(portKey(S.hostCam), label || "");
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
    const rem = LS.get(portKey(S.hostCam), undefined);
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
  LS.set(`cvcal:orient:${S.slug}`, S.orient);
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

$("oRotate").addEventListener("change", (e) => setOrientation({ rotate: +e.target.value }));

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
    const badge = $("shotBadge");
    badge.textContent = det ? `✔ board detected (${det.n} corners)`
                            : "✖ no board — discarded";
    badge.className = "shot-badge " + (det ? "good" : "bad");
    setTimeout(() => badge.classList.add("hidden"), 1500);
    if (det) {
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
    const remembered = LS.get(portKey(cam), undefined);
    let sel = fam[0] || null;
    if (remembered !== undefined) {
      const f = fam.find((x) => (x.label || "") === remembered);
      if (f) sel = f;
    }
    const tr = document.createElement("tr");
    const famOpts = fam.map((f) =>
      `<option value="${esc(f.label || "")}"${f === sel ? " selected" : ""}>` +
      `${esc(displayLabel(f.label))}</option>`).join("") +
      '<option value="__new">➕ Create new calibration…</option>';
    tr.innerHTML = `
      <td><a class="camlink">${esc(cam.name)}</a>
          <div class="dim small">/dev/video${cam.node}</div></td>
      <td>${esc(cam.usb?.id_vendor || "?")}:${esc(cam.usb?.id_product || "?")}
          <div class="dim small">serial ${esc(cam.usb?.serial || "none")}</div>
          ${cam.duplicate
            ? '<div><span class="badge warn">duplicate ID — use labels</span></div>'
            : cam.serial_trusted ? "" : '<div><span class="badge warn">generic serial</span></div>'}</td>
      <td class="cal-cell">
        <select class="verSel">${famOpts}</select>
        <div class="calinfo dim small"></div>
        <button class="btn small success calibBtn">Calibrate</button>
      </td>
      <td class="live-cell"><img class="grid-live" alt="">
          <div class="live-cell-bar">
            <button class="btn small rotBtn"
              title="Rotate this camera's view 90° clockwise — always saved (with its calibration when one is selected)">⟳ 90°</button>
            <span class="thumb-note"><span class="spin">◐</span> connecting…</span>
          </div></td>`;
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
      LS.set(portKey(cam), verSel.value);
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
  // the camera's saved view rotation applies even before any calibration
  const lsRot = LS.get(`cvcal:orient:${row.cam.slug}`, {}).rotate || 0;
  if (!row.sel) {
    info.innerHTML = '<span class="badge warn">not calibrated</span>';
    btn.textContent = "Calibrate";
    row.rot = lsRot;
  } else {
    const cal = await fetchCal(row.sel.slug);
    if (calValid(cal)) {
      const i = cal.intrinsic;
      info.textContent = `RMS ${i.rms_reprojection_error_px?.toFixed(3)} px ` +
                         `@ ${i.image_size?.join("×")}`;
      btn.textContent = "Recalibrate";
    } else {
      info.innerHTML = '<span class="badge warn">uncalibrated</span> — no data yet';
      btn.textContent = "Calibrate";
    }
    row.rot = cal?.extrinsic?.orientation?.rotate_deg_cw ?? lsRot;
  }
  applyGridRotation(img, row.rot);
}

/* Rotation is cheap to change and expensive to lose: every adjustment is
   saved — to the browser's per-camera key always, into the selected
   calibration file when there is one, and mirrored to the collect tab if
   this camera is selected there. */
async function rotateCam(row) {
  row.rot = ((row.rot || 0) + 90) % 360;
  applyGridRotation(row.tr.querySelector("img.grid-live"), row.rot);
  LS.set(`cvcal:orient:${row.cam.slug}`, { rotate: row.rot });
  if (S.slug === row.cam.slug) {          // sync the collect/measure tabs
    S.orient.rotate = row.rot;
    updateOrientationUI();
    applyOrientationCss();
  }
  if (!row.sel) {
    toast(`Rotation ${row.rot}° saved for this camera (no calibration file yet).`);
    return;
  }
  const cal = await fetchCal(row.sel.slug);
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
  LS.set(portKey(cam), label || "");
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
  LS.set(portKey(cam), label);
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

function renderCalFiles(cals) {
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
      tr.innerHTML = `
        <td><b>${esc(displayLabel(ver.label))}</b></td>
        <td>${devs.length ? esc(devs.join(", ")) : '<span class="dim">—</span>'}</td>
        <td class="ver-status dim small">…</td>
        <td class="ver-acts">
          <button class="btn small" title="Rename label">✎</button>
          <button class="btn small danger-outline" title="Delete calibration file">🗑</button>
        </td>`;
      const [renameBtn, delBtn] = tr.querySelectorAll("button");
      renameBtn.addEventListener("click", () => renameCalFlow(base, ver));
      delBtn.addEventListener("click", () => deleteCalFlow(base, ver));
      fetchCal(ver.slug).then((cal) => {
        const st = tr.querySelector(".ver-status");
        if (calValid(cal)) {
          st.textContent = `RMS ${cal.intrinsic.rms_reprojection_error_px?.toFixed(3)} px`;
        } else {
          st.innerHTML = '<span class="badge warn">uncalibrated</span>';
        }
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
    if (row.sel?.slug === ver.slug) LS.set(portKey(row.cam), label);
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
      <div style="font:9pt sans-serif;margin-top:1.5mm">ID ${t.id} — ${t.mm} mm — 36h11</div>
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
  }
});

/* -------------------------------------------------------------------- init */
restoreForm();
(async () => {
  await detectHost();
  if (HOST) {
    document.title += " — local (host cameras)";
    $("configTabBtn").classList.remove("hidden");
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
