"""Host-side camera discovery and capture for local (lab) mode.

Ported from the original uvc_camera_cal tool: enumerates UVC capture nodes
via /sys/class/video4linux + linuxpy V4L2 ioctls, pulls USB descriptors
(vendor/product/serial) from sysfs, and owns the cameras exclusively so
the browser never has to.
"""

import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import cv2

try:
    from linuxpy.video.device import Capability
    from linuxpy.video.device import Device as V4L2Device
    HAVE_LINUXPY = True
except ImportError:
    HAVE_LINUXPY = False

SYS_V4L = Path("/sys/class/video4linux")
BY_ID_DIR = Path("/dev/v4l/by-id")

# serials seen on clone hardware that are not unique per unit
KNOWN_BAD_SERIALS = {
    "", "0", "1", "01", "0001", "00000000", "000000000000", "1234567890",
    "12345", "sn001", "sn0001", "serial", "default", "01.00.00", "none",
    "usb camera", "0x0001",
}


def _read_sys(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _usb_device_dir(video_name):
    try:
        d = (SYS_V4L / video_name / "device").resolve()
    except OSError:
        return None
    while d != Path("/"):
        if (d / "idVendor").exists():
            return d
        d = d.parent
    return None


def _has_audio_interface(usb_dev_dir):
    if usb_dev_dir is None:
        return False
    for cls_file in usb_dev_dir.glob("*/bInterfaceClass"):
        if _read_sys(cls_file) == "01":
            return True
    return False


def _by_id_map():
    links = {}
    if BY_ID_DIR.is_dir():
        for link in sorted(BY_ID_DIR.iterdir()):
            links[os.path.realpath(link)] = str(link)
    return links


def _serial_trusted(serial):
    return bool(serial) and serial.strip().lower() not in KNOWN_BAD_SERIALS


def slug_for(cam):
    """Canonical identifier: always vid_pid_serial (udev-independent).

    The /dev/v4l/by-id name is NOT used: with two identical cameras udev can
    only point the by-id symlink at one of them, so by-id-derived IDs differ
    between identical units and between replugs. VID:PID+serial reads the
    same on every machine, every time."""
    usb = cam.get("usb", {})
    tail = usb.get("serial") or usb.get("product") or cam["name"]
    base = "usb-" + "_".join([usb.get("id_vendor") or "unk",
                              usb.get("id_product") or "unk", str(tail)])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def _is_capture_node(dev_path):
    if not HAVE_LINUXPY:
        name = Path(dev_path).name
        return _read_sys(SYS_V4L / name / "index") == "0"
    try:
        with V4L2Device(dev_path) as d:
            return bool(d.info.device_capabilities & Capability.VIDEO_CAPTURE)
    except Exception:
        return False


def list_cameras():
    by_id = _by_id_map()
    cams = []
    if not SYS_V4L.is_dir():
        return cams
    for entry in sorted(SYS_V4L.iterdir(), key=lambda p: p.name):
        m = re.match(r"video(\d+)$", entry.name)
        if not m:
            continue
        dev_path = f"/dev/{entry.name}"
        if not _is_capture_node(dev_path):
            continue
        usb_dir = _usb_device_dir(entry.name)
        usb = {}
        if usb_dir is not None:
            usb = {
                "id_vendor": _read_sys(usb_dir / "idVendor"),
                "id_product": _read_sys(usb_dir / "idProduct"),
                "manufacturer": _read_sys(usb_dir / "manufacturer"),
                "product": _read_sys(usb_dir / "product"),
                "serial": _read_sys(usb_dir / "serial"),
                "usb_version": _read_sys(usb_dir / "version"),
                "speed_mbps": _read_sys(usb_dir / "speed"),
                "bus_path": usb_dir.name,
            }
        cam = {
            "node": int(m.group(1)),
            "path": dev_path,
            "name": _read_sys(entry / "name") or entry.name,
            "by_id": by_id.get(os.path.realpath(dev_path)),
            "usb": usb,
            "has_microphone": _has_audio_interface(usb_dir),
            "serial_trusted": _serial_trusted(usb.get("serial")),
        }
        cam["slug"] = slug_for(cam)
        cams.append(cam)
    # duplicate detection: same ID on two connected cameras proves clone
    # serials. They intentionally KEEP the same base ID (one calibration
    # family, distinguished by labels) — a port-based ID would not survive
    # replugging or another machine.
    slug_counts = {}
    for c in cams:
        slug_counts[c["slug"]] = slug_counts.get(c["slug"], 0) + 1
    for c in cams:
        if slug_counts[c["slug"]] > 1:
            c["serial_trusted"] = False
            c["duplicate"] = True
    return cams


SYS_USB = Path("/sys/bus/usb/devices")


def list_usb_tree():
    """Flat list of USB devices with parent links, for the topology view.

    Keys are kernel names: roots are "usbN" (one per bus; an xHCI
    controller exposes separate USB2 and USB3 buses, so buses are grouped
    by their PCI controller address), devices are "1-1.4"-style port
    paths. Interface entries ("1-1:1.0") are folded into their device as
    has_video / has_audio / is_hub flags."""
    devices = []
    if not SYS_USB.is_dir():
        return devices
    for entry in sorted(SYS_USB.iterdir()):
        name = entry.name
        if ":" in name:                       # interface, not a device
            continue
        if name.startswith("usb") and name[3:].isdigit():
            is_root, bus = True, int(name[3:])
        elif re.match(r"^\d+-[\d.]+$", name):
            is_root, bus = False, int(name.split("-")[0])
        else:
            continue
        is_hub = _read_sys(entry / "bDeviceClass") == "09"
        has_video = has_audio = False
        video_devs = []
        for intf in entry.glob(f"{name}:*"):
            cls = _read_sys(intf / "bInterfaceClass")
            if cls == "0e":
                has_video = True
            elif cls == "01":
                has_audio = True
            elif cls == "09":
                is_hub = True
            v4l = intf / "video4linux"
            if v4l.is_dir():
                for node in sorted(v4l.iterdir()):
                    # capture nodes only (index 0), not metadata nodes
                    if _read_sys(node / "index") == "0":
                        video_devs.append("/dev/" + node.name)
        if is_root:
            parent = None
            # .../pci0000:00/.../0000:0c:00.3/usb1 -> PCI controller addr
            controller = os.path.realpath(entry).rstrip("/").split("/")[-2]
        else:
            tail = name.split("-", 1)[1]
            parent = name.rsplit(".", 1)[0] if "." in tail else f"usb{bus}"
            controller = None                 # filled from the bus root
        devices.append({
            "key": name, "bus": bus, "parent": parent, "is_root": is_root,
            "is_hub": is_hub, "has_video": has_video, "has_audio": has_audio,
            "video_devs": video_devs,
            "vid": _read_sys(entry / "idVendor"),
            "pid": _read_sys(entry / "idProduct"),
            "product": _read_sys(entry / "product"),
            "manufacturer": _read_sys(entry / "manufacturer"),
            "serial": _read_sys(entry / "serial"),
            "speed_mbps": _read_sys(entry / "speed"),
            "usb_version": (_read_sys(entry / "version") or "").strip(),
            "controller": controller,
        })
    roots = {d["key"]: d for d in devices if d["is_root"]}
    for d in devices:
        if not d["is_root"]:
            root = roots.get(f"usb{d['bus']}")
            d["controller"] = root["controller"] if root else None
    return devices


def camera_modes(dev_path):
    modes = []
    if not HAVE_LINUXPY:
        return {"modes": modes, "driver": {}}
    try:
        with V4L2Device(dev_path) as d:
            info = d.info
            driver = {"driver": str(info.driver), "card": str(info.card),
                      "bus_info": str(info.bus_info)}
            seen = set()
            for fs in info.frame_sizes():
                fmt = fs.pixel_format.name
                w, h = fs.info.width, fs.info.height
                key = (fmt, w, h)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    fps = sorted({float(ft.max_fps) for ft in
                                  info.fps_intervals(fs.pixel_format, w, h)},
                                 reverse=True)
                except Exception:
                    fps = []
                modes.append({"format": fmt, "width": w, "height": h,
                              "fps": fps})
            modes.sort(key=lambda x: (x["format"], -x["width"] * x["height"]))
            return {"modes": modes, "driver": driver}
    except Exception as e:
        return {"modes": [], "driver": {}, "error": str(e)}


class CameraStream:
    """Single shared capture stream; readers pull the latest frame."""

    def __init__(self):
        self._cap = None
        self._thread = None
        self._running = False
        self._cond = threading.Condition()
        self._frame = None
        self._frame_ts = 0.0
        self._seq = 0
        self._times = None      # recent frame timestamps for fps_actual
        self.info = {}
        # why the capture loop gave up, kept after the stream dies so the
        # client can say what went wrong instead of just showing no frames
        self.error = None

    @property
    def fps_actual(self):
        t = self._times
        if not t or len(t) < 2:
            return 0.0
        span = t[-1] - t[0]
        return round((len(t) - 1) / span, 2) if span > 0 else 0.0

    def _open(self, path, width, height, fps):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def start(self, cam, width, height, fps=0):
        self.stop()
        cap = self._open(cam["path"], width, height, fps)
        if cap is None:
            raise RuntimeError(
                f"Could not open {cam['path']} — in use by another program?")
        self._path = cam["path"]
        self._settings = (width, height, fps)
        self._times = deque(maxlen=60)
        self._cap = cap
        self._running = True
        self.error = None
        self.info = {
            "slug": cam["slug"], "node": cam["node"], "name": cam["name"],
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        }
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return dict(self.info)

    def _loop(self):
        fails = 0
        reopens = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                time.sleep(0.05)
                if fails >= 40:      # ~2s of dead reads: wedged or unplugged
                    # cap retries: a device that reopens but can never
                    # stream (e.g. USB bandwidth refused) must not be
                    # retried forever — it spams the kernel log every
                    # cycle and never recovers on its own
                    if reopens < 5 and os.path.exists(self._path):
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        cap = self._open(self._path, *self._settings)
                        if cap is not None:
                            self._cap = cap
                            fails = 0
                            reopens += 1
                            continue
                    # device is gone: END the stream instead of freezing on
                    # the last frame — clients see the connection close and
                    # can react, rather than silently grabbing a stale image
                    self._running = False
                    # A camera that opens but never yields a frame is almost
                    # always the USB controller refusing the isochronous
                    # bandwidth at stream-on (open succeeds, VIDIOC_STREAMON
                    # doesn't). Say so: otherwise the tile just sits at
                    # "warming up…" with nothing to act on.
                    w, h, _fps = self._settings
                    self.error = (
                        f"{self._path} disappeared — camera unplugged"
                        if not os.path.exists(self._path) else
                        f"opened, but delivered no frames at {w}x{h} — the USB "
                        "controller refused the stream, which means the bus is "
                        "out of bandwidth. Use a lower resolution, or move this "
                        "camera to a port on a different USB controller.")
                    # Release the handle before giving up. Holding it keeps the
                    # fd open AND keeps the device's USB bandwidth reserved, so
                    # a camera that failed once could never be reopened -- not
                    # by a retry, not by another process -- and it starved its
                    # bus-mates for as long as the server lived.
                    try:
                        if self._cap is not None:
                            self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    with self._cond:
                        self._cond.notify_all()
                    return
                continue
            fails = 0
            reopens = 0          # real frames: future stalls retry afresh
            now = time.time()    # closest we can get to capture time
            if self._times is not None:
                self._times.append(now)
            with self._cond:
                self._frame = frame
                self._frame_ts = now
                self._seq += 1
                self._cond.notify_all()

    def get_frame(self, last_seq=0, timeout=2.0):
        """-> (frame, seq, captured_at). The timestamp matters for anything
        fusing several cameras: these are free-running USB cameras at
        different rates, so 'the newest frame' from two of them can be tens
        of milliseconds apart, and triangulating across that gap turns
        object motion into position error."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._frame is None:
                return None, last_seq, 0.0
            return self._frame.copy(), self._seq, self._frame_ts

    @property
    def running(self):
        return self._running and self._frame is not None

    @property
    def started(self):
        """True once start() succeeded, even before the first frame lands."""
        return self._running

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._cond:
            self._frame = None
            self._frame_ts = 0.0
            self._seq = 0
        self.info = {}


def focus_info(dev_path):
    """V4L2 focus state for a device, if it exposes focus controls.
    (Cameras like the OAK's UVC mode expose none — their focus is set by
    whatever booted them.)"""
    out = {"supported": False}
    if not HAVE_LINUXPY:
        return out
    try:
        with V4L2Device(dev_path) as d:
            for c in d.controls.values():
                name = getattr(c, "config_name", "") or ""
                if name == "focus_absolute":
                    out.update({
                        "supported": True,
                        "value": int(c.value),
                        "min": int(getattr(c, "minimum", 0)),
                        "max": int(getattr(c, "maximum", 255)),
                        "step": int(getattr(c, "step", 1) or 1),
                    })
                elif name in ("focus_automatic_continuous", "focus_auto"):
                    out["auto_supported"] = True
                    out["auto"] = bool(c.value)
    except Exception as e:
        out["error"] = str(e)
    return out


def focus_set(dev_path, value=None, auto=None):
    """Set manual focus and/or toggle autofocus; returns the new state."""
    if not HAVE_LINUXPY:
        return {"supported": False, "error": "linuxpy unavailable"}
    try:
        with V4L2Device(dev_path) as d:
            # order matters: drivers reject focus_absolute while AF is on
            if auto is not None:
                for c in d.controls.values():
                    if (getattr(c, "config_name", "") or "") in (
                            "focus_automatic_continuous", "focus_auto"):
                        c.value = 1 if auto else 0
            if value is not None:
                for c in d.controls.values():
                    if (getattr(c, "config_name", "") or "") == "focus_absolute":
                        c.value = int(value)
    except Exception as e:
        info = focus_info(dev_path)
        info["error"] = str(e)
        return info
    return focus_info(dev_path)
