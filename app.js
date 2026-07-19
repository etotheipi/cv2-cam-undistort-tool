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
const S = {
  pyReady: false,
  devices: [],
  device: null,          // selected MediaDeviceInfo
  slug: null,
  stream: null,
  trackSettings: {},
  trackCaps: null,
  micPresent: false,
  collecting: false,
  collectTimers: [],
  images: [],            // records for current camera (from localStorage)
  lastResult: null,      // last calibration result (full JSON object)
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

function settings() {
  return {
    cols: parseInt($("cols").value, 10),
    rows: parseInt($("rows").value, 10),
    square_size: parseFloat($("squareSize").value),
    units: $("units").value,
    name: $("camName").value.trim(),
  };
}
function measureSettings() {
  return {
    cols: parseInt($("mCols").value, 10),
    rows: parseInt($("mRows").value, 10),
    square_size: parseFloat($("mSquareSize").value),
    units: $("mUnits").value,
  };
}

function saveForm() {
  LS.set("cvcal:form", {
    cols: $("cols").value, rows: $("rows").value,
    square: $("squareSize").value, units: $("units").value,
  });
  if (S.slug) LS.set(`cvcal:name:${S.slug}`, $("camName").value);
}
function restoreForm() {
  const f = LS.get("cvcal:form", {});
  if (f.cols) $("cols").value = f.cols;
  if (f.rows) $("rows").value = f.rows;
  if (f.square) $("squareSize").value = f.square;
  if (f.units) $("units").value = f.units;
  prevUnit.units = $("units").value;
}
["cols", "rows", "squareSize", "units", "camName"].forEach((id) =>
  $(id).addEventListener("change", saveForm));

/* Changing a units dropdown converts the paired numeric value so the
   physical size stays the same (25 mm -> 0.98425197 in). */
function fmtNum(v) { return +v.toPrecision(8); }
const prevUnit = { units: $("units").value, mUnits: $("mUnits").value };
function convertUnitField(selectId, numId) {
  const nu = $(selectId).value, ou = prevUnit[selectId];
  const v = parseFloat($(numId).value);
  if (v && nu !== ou) {
    $(numId).value = fmtNum(v * UNIT_TO_MM[ou] / UNIT_TO_MM[nu]);
  }
  prevUnit[selectId] = nu;
}
$("units").addEventListener("change", () => { convertUnitField("units", "squareSize"); saveForm(); });

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
    const src = await (await fetch("py/calib_core.py")).text();
    pyodide.FS.writeFile("/home/pyodide/calib_core.py", src);
    pyodide.runPython("import calib_core");
    py = pyodide.globals.get("calib_core");
    S.pyReady = true;
    setBoot(`ready — Python + OpenCV ${py.cv2_version()}`, "ready");
    if (S.activeCal) pushActiveCalToPython();
    updateButtons();
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
  activeTab = name;
  document.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach((p) =>
    p.classList.toggle("active", p.id === "tab-" + name));
}

/* ----------------------------------------------------------- camera setup */
async function refreshDevices() {
  const devs = await navigator.mediaDevices.enumerateDevices();
  S.devices = devs.filter((d) => d.kind === "videoinput");
  const havePermission = S.devices.some((d) => d.label);
  $("grantBtn").style.display = havePermission ? "none" : "";
  const sel = $("cameraSelect");
  const prev = sel.value;
  sel.innerHTML = '<option value="">— select a camera —</option>';
  for (const d of S.devices) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    const slug = slugify(d.label);
    opt.textContent = (d.label || `camera ${sel.length}`) +
      (LS.cal(slug) ? "  ✔ calibrated" : "");
    sel.appendChild(opt);
  }
  if (prev) sel.value = prev;
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

const RES_PRESETS = [
  [640, 480], [800, 600], [1024, 768], [1280, 720], [1600, 896],
  [1920, 1080], [2560, 1440], [3840, 2160],
];

async function selectCamera(deviceId, width, height) {
  stopCollecting(true);
  const dev = S.devices.find((d) => d.deviceId === deviceId);
  if (!dev) return;
  S.device = dev;
  S.slug = slugify(dev.label);
  await openStream(width || 1280, height || 720);
  if (!S.stream) return;
  populateModes();
  renderCameraInfo();
  $("camName").value = LS.get(`cvcal:name:${S.slug}`, "") || dev.label || "";
  loadImages();
  // camera's stored calibration becomes active unless a file was uploaded
  if (S.activeCalSource !== "uploaded file") {
    const cal = LS.cal(S.slug);
    if (cal) activateCalibration(cal, "storage");
    else deactivateCalibration();
  }
  updateButtons();
}

async function openStream(width, height) {
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: S.device.deviceId },
        width: { ideal: width }, height: { ideal: height },
      },
    });
  } catch (e) {
    toast("Could not open camera: " + e.message, true);
    renderCameraInfo();
    return;
  }
  const track = S.stream.getVideoTracks()[0];
  S.trackSettings = track.getSettings();
  S.trackCaps = track.getCapabilities ? track.getCapabilities() : null;
  const audio = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.kind === "audioinput");
  S.micPresent = !!S.trackSettings.groupId &&
    audio.some((a) => a.groupId === S.trackSettings.groupId);
  $("liveVideo").srcObject = S.stream;
  $("measVideo").srcObject = S.stream;
  $("liveOverlayMsg").classList.add("hidden");
}

function populateModes() {
  const sel = $("modeSelect");
  sel.innerHTML = "";
  const caps = S.trackCaps;
  const maxW = caps?.width?.max || 1920;
  const maxH = caps?.height?.max || 1080;
  const list = RES_PRESETS.filter(([w, h]) => w <= maxW && h <= maxH);
  if (!list.some(([w, h]) => w === maxW && h === maxH)) list.push([maxW, maxH]);
  for (const [w, h] of list) {
    const opt = document.createElement("option");
    opt.value = `${w}x${h}`;
    opt.textContent = `${w} × ${h}` + (w === maxW && h === maxH ? " (max)" : "");
    sel.appendChild(opt);
  }
  const cur = `${S.trackSettings.width}x${S.trackSettings.height}`;
  if (![...sel.options].some((o) => o.value === cur)) {
    const opt = document.createElement("option");
    opt.value = cur;
    opt.textContent = cur.replace("x", " × ") + " (current)";
    sel.appendChild(opt);
  }
  sel.value = cur;
  sel.disabled = false;
}

$("modeSelect").addEventListener("change", async (e) => {
  const [w, h] = e.target.value.split("x").map(Number);
  stopCollecting(true);
  await openStream(w, h);
  renderCameraInfo();
});

function renderCameraInfo() {
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
  rows.push(["Note", '<span class="dim">Browsers cannot read USB serial numbers or full mode lists — use the name field to distinguish identical cameras.</span>', true]);
  $("cameraInfo").innerHTML = "<table>" + rows.map(([k, v, raw]) =>
    `<tr><td>${k}</td><td>${raw ? v : esc(v)}</td></tr>`).join("") + "</table>";
}

/* ------------------------------------------------------------ frame grab */
const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });
function grabFrame(video) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return null;
  grabCanvas.width = w; grabCanvas.height = h;
  grabCtx.drawImage(video, 0, 0);
  return grabCtx.getImageData(0, 0, w, h);
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
  const ready = !!S.stream && S.pyReady;
  $("collectBtn").disabled = !ready;
  $("clearBtn").disabled = !S.slug || (S.images.length === 0 && !S.collecting);
  $("calibrateBtn").disabled = !ready || S.images.length < 5 || S.collecting;
  $("snapRawBtn").disabled = !ready;
  $("snapUndBtn").disabled = !ready || !S.activeCal;
  $("downloadCalBtn").disabled = !S.lastResult;
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
  if (!S.stream || !S.pyReady) return;
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

let snapInFlight = false;
async function snapCalibImage(manual = false) {
  if (snapInFlight || !S.pyReady || (!manual && !S.collecting)) return;
  snapInFlight = true;
  try {
    const im = grabFrame($("liveVideo"));
    if (!im) return;
    if (S.images.length &&
        (S.images[0].w !== im.width || S.images[0].h !== im.height)) {
      toast(`Resolution changed (collection is ${S.images[0].w}×${S.images[0].h}, ` +
            `stream is ${im.width}×${im.height}) — clear all or switch back.`, true);
      stopCollecting();
      return;
    }
    const { cols, rows } = settings();
    const corners = JSON.parse(py.detect_corners(im.data, im.width, im.height, cols, rows));
    const fl = $("flash");
    fl.classList.remove("on");
    void fl.offsetWidth;
    fl.classList.add("on");
    const badge = $("shotBadge");
    badge.textContent = corners ? "✔ board detected" : "✖ no board — discarded";
    badge.className = "shot-badge " + (corners ? "good" : "bad");
    setTimeout(() => badge.classList.add("hidden"), 1500);
    if (corners) {
      const rec = {
        id: Date.now() + "-" + Math.random().toString(36).slice(2, 6),
        ts: new Date().toISOString(),
        w: im.width, h: im.height,
        cols, rows, corners,
        thumb: makeThumb(im),
      };
      S.images.unshift(rec);               // newest first
      if (LS.setImages(S.slug, S.images)) {
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
  S.images = LS.images(S.slug);
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
    LS.setImages(S.slug, S.images);
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
  LS.setImages(S.slug, []);
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
  if (!st.square_size || !st.cols || !st.rows) {
    toast("Set square size and inner-corner counts first.", true);
    return;
  }
  switchTab("results");
  resetSteps();
  try {
    // gather -----------------------------------------------------------
    setStep("gather", "active", "");
    await tick();
    const usable = S.images.filter((r) =>
      r.cols === st.cols && r.rows === st.rows &&
      r.w === S.images[0].w && r.h === S.images[0].h);
    const skipped = S.images.length - usable.length;
    log(`${S.images.length} stored images; ${usable.length} match ` +
        `${st.cols}×${st.rows} corners @ ${S.images[0].w}×${S.images[0].h}` +
        (skipped ? ` (${skipped} skipped — different board/resolution)` : ""));
    if (usable.length < 5) throw new Error(
      `Only ${usable.length} usable views — collect more, or check the ` +
      `inner-corner settings match what was used during collection.`);
    // oldest-first for stable indexing in results
    const ordered = [...usable].reverse();
    setStep("gather", "done", `${usable.length} views`);

    // solve --------------------------------------------------------------
    setStep("solve", "active", "this can take a few seconds…");
    log(`Calibrating at ${ordered[0].w}×${ordered[0].h} with ${ordered.length} views…`);
    await tick();
    const res = JSON.parse(py.calibrate(
      JSON.stringify(ordered.map((r) => r.corners)),
      ordered[0].w, ordered[0].h, st.cols, st.rows, st.square_size));
    if (res.error) throw new Error(res.error);
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
  return {
    schema_version: 1,
    name: st.name || S.device.label || S.slug,
    slug: S.slug,
    camera: {
      platform: "web",
      label: S.device.label,
      usb_vid_pid: (S.device.label.match(/\(([0-9a-f]{4}:[0-9a-f]{4})\)/i) || [])[1] || null,
      device_id: S.device.deviceId,
      group_id: S.device.groupId,
      microphone: S.micPresent,
      capture_settings: {
        width: S.trackSettings.width, height: S.trackSettings.height,
        frame_rate: S.trackSettings.frameRate,
      },
      user_agent: navigator.userAgent,
    },
    intrinsic: {
      calibrated_at: new Date().toISOString(),
      image_size: res.image_size,
      camera_matrix: res.camera_matrix,
      dist_coeffs: res.dist_coeffs,
      distortion_model: "opencv_plumb_bob",
      rms_reprojection_error_px: res.rms,
      per_view_errors_px: res.per_view.map((e, i) => ({
        index: i, ts: ordered[i]?.ts, error_px: e })),
      num_images: res.per_view.length,
      checkerboard: {
        inner_corners: [st.cols, st.rows],
        square_size: st.square_size,
        units: st.units,
        square_size_mm: st.square_size * UNIT_TO_MM[st.units],
      },
    },
    extrinsic: existing?.extrinsic || {},
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
      <tr><td class="dim">Images used&nbsp;</td><td>${res.per_view.length} @ ${res.image_size.join("×")}</td></tr>
      <tr><td class="dim">Stored as&nbsp;</td><td><code>${esc(cal.slug)}</code> (browser localStorage)</td></tr>
    </table>`;
}

function renderErrChart(res, ordered) {
  const maxErr = Math.max(...res.per_view, 0.5);
  $("errChart").innerHTML = res.per_view.map((e, i) => {
    const cls = e > 1 ? "poor" : e > 0.5 ? "okay" : "";
    const label = (ordered[i]?.ts || `view ${i}`).slice(5, 19).replace("T", " ");
    return `<div class="err-row">
      <span class="fname">#${i + 1} — ${esc(label)}</span>
      <span class="bar-track"><span class="bar ${cls}" style="width:${(100 * e / maxErr).toFixed(1)}%"></span></span>
      <span>${e.toFixed(3)}</span></div>`;
  }).join("");
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
  $("activeCalInfo").innerHTML =
    `<span class="badge ok">active</span> ${esc(cal.name || cal.slug)} ` +
    `<span class="dim">(${source}) — RMS ${i.rms_reprojection_error_px?.toFixed(3)} px ` +
    `@ ${i.image_size?.join("×")}</span>`;
  const cb = i.checkerboard;
  if (cb) {
    $("mSquareSize").value = cb.square_size;
    $("mUnits").value = cb.units;
    $("mCols").value = cb.inner_corners[0];
    $("mRows").value = cb.inner_corners[1];
    prevUnit.mUnits = cb.units;
  }
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
  if (activeTab !== "measure" || !S.activeCal || !S.pyReady || !S.stream || undBusy) return;
  undBusy = true;
  try {
    const im = grabFrame($("measVideo"));
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
  if (!S.pyReady || !S.stream) return;
  let im;
  if (source === "undistorted") {
    if (!undCanvas.width) { toast("Undistorted view not ready yet.", true); return; }
    im = undCtx.getImageData(0, 0, undCanvas.width, undCanvas.height);
  } else {
    im = grabFrame($("measVideo"));
  }
  if (!im) { toast("No frame available.", true); return; }
  S.snap = { imageData: im, w: im.width, h: im.height, source };
  S.measurements = [];
  S.measurePts = [];
  setMeasuring(false);
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
    m.cols, m.rows, m.square_size,
    S.snap.source === "undistorted"));
  S.snapBoard = res.found ? res.corners : null;
  $("snapLabel").textContent =
    `${S.snap.source} frame — ${S.snap.w}×${S.snap.h} — ` +
    (res.found ? `checkerboard detected (${m.cols}×${m.rows})` : "no checkerboard");
  $("snapMeasureBtn").disabled = !res.found;
  if (!res.found && S.measuring) setMeasuring(false);
  drawMeasureOverlay();
}

/* Board-param changes while a snap is open: re-fit the homography and
   recompute every stored measurement from its original click points, so
   existing annotations update in place (e.g. after a unit switch). */
function remeasureAll() {
  const pairs = S.measurements.map((m) => [m.p1, m.p2]);
  S.measurements = [];
  if (S.snapBoard) {
    const m = measureSettings();
    for (const [p1, p2] of pairs) {
      const res = JSON.parse(py.measure_points(p1[0], p1[1], p2[0], p2[1]));
      if (res.error) continue;
      const mm = res.distance * UNIT_TO_MM[m.units];
      S.measurements.push({
        p1, p2, units: m.units,
        distance: +res.distance.toFixed(3),
        distance_mm: +mm.toFixed(2),
        distance_in: +(mm / 25.4).toFixed(3),
      });
    }
  }
  drawMeasureOverlay();
  renderMeasureList();
}

["mCols", "mRows", "mSquareSize", "mUnits"].forEach((id) =>
  $(id).addEventListener("change", () => {
    if (id === "mUnits") convertUnitField("mUnits", "mSquareSize");
    if (S.snap) {
      S.measurePts = [];
      prepareBoard();
      remeasureAll();
    }
  }));

$("snapViewBtn").addEventListener("click", () => {
  if (S.snap) openLightbox(compositeDataURL());
});
$("snapMeasureBtn").addEventListener("click", () => setMeasuring(!S.measuring));

function setMeasuring(on) {
  S.measuring = on;
  S.measurePts = [];
  $("snapMeasureBtn").classList.toggle("active-mode", on);
  $("snapWrap").classList.toggle("measuring", on);
  $("measureHint").textContent = on ? "Click two points on the checkerboard plane…" : "";
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
    const m = measureSettings();
    const res = JSON.parse(py.measure_points(p1[0], p1[1], p2[0], p2[1]));
    if (res.error) { toast(res.error, true); drawMeasureOverlay(); return; }
    const mm = res.distance * UNIT_TO_MM[m.units];
    S.measurements.push({
      p1, p2, units: m.units,
      distance: +res.distance.toFixed(3),
      distance_mm: +mm.toFixed(2),
      distance_in: +(mm / 25.4).toFixed(3),
    });
    drawMeasureOverlay();
    renderMeasureList();
    $("measureHint").textContent =
      `${S.measurements.at(-1).distance} ${m.units} — click two more points, or toggle 📏 to finish.`;
  }
});

function renderMeasureList() {
  const fmtAlt = (m) => (m.units === "in" || m.units === "ft")
    ? `${m.distance_mm} mm` : `${m.distance_in} in`;
  $("measureList").innerHTML = S.measurements.map((m, i) =>
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

function drawMeasureOverlay() {
  const c = $("snapCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const s = Math.max(c.width / 1280, 0.7);
  if (S.measuring && S.snapBoard) {
    ctx.fillStyle = "rgba(80,200,120,0.6)";
    for (const [x, y] of S.snapBoard) {
      ctx.beginPath(); ctx.arc(x, y, 2.5 * s, 0, Math.PI * 2); ctx.fill();
    }
  }
  const drawPt = (p, color = "#ff5a00") => {
    ctx.beginPath(); ctx.arc(p[0], p[1], 6 * s, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5 * s; ctx.strokeStyle = "#fff"; ctx.stroke();
  };
  for (const m of S.measurements) {
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
  for (const p of S.measurePts) drawPt(p, "#4f9cf7");
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
  $("snapPanel").style.display = "none";
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
    else if (S.stream && S.pyReady) snapCalibImage(true);
  } else if (activeTab === "measure") {
    e.preventDefault();
    if (S.stream && S.pyReady) takeSnap("raw");
  }
});

/* -------------------------------------------------------------------- init */
restoreForm();
if (!navigator.mediaDevices?.getUserMedia) {
  toast("This browser does not support camera access (getUserMedia).", true);
} else {
  refreshDevices();
  navigator.mediaDevices.addEventListener?.("devicechange", refreshDevices);
}
bootPython();
