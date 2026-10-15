"""The optional uvcvideo bandwidth patch (extras/uvc-mjpeg-bandwidth), driven
from the USB Device Tree tab: status, prerequisites, build, load, restore.

Building runs as the bridge's own user. Swapping a loaded kernel module
needs root, so load/restore run one fixed command through `sudo -n` when
that works without a password, else `pkexec` (the desktop's polkit password
prompt), and otherwise hand back the command to run in a terminal. Nothing
here installs anything permanently; the tab shows those steps as text.
"""

import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import threading
import time

EXTRA = pathlib.Path(__file__).resolve().parent.parent / "extras" / "uvc-mjpeg-bandwidth"
BUILT_KO = EXTRA / "build" / "uvcvideo.ko"
PARAMS = pathlib.Path("/sys/module/uvcvideo/parameters")
CONF = pathlib.Path("/etc/modprobe.d/uvcvideo.conf")
DEFAULT_CAP = 1600          # bytes per 125 us: 3 cameras per strict USB 2 bus
SYS_USB = pathlib.Path("/sys/bus/usb/devices")


def _read(p):
    try:
        return pathlib.Path(p).read_text().strip()
    except OSError:
        return ""


def _run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def _distro():
    txt = _read("/etc/os-release")
    ids = " ".join(re.findall(r'^(?:ID|ID_LIKE)="?([^"\n]+)', txt, re.M)).lower()
    return ids


def install_hint(kver):
    ids = _distro()
    if any(k in ids for k in ("debian", "ubuntu", "mint")):
        return f"sudo apt install git build-essential linux-headers-{kver}"
    if any(k in ids for k in ("fedora", "rhel", "centos")):
        return f"sudo dnf install git make gcc kernel-devel-{kver}"
    if "arch" in ids:
        return "sudo pacman -S git base-devel linux-headers"
    return f"install git, make, gcc and the kernel headers for {kver}"


def _vermagic(path):
    rc, out = _run(["modinfo", "-F", "vermagic", str(path)])
    return out.split()[0] if rc == 0 and out else None


def _privilege():
    if os.geteuid() == 0:
        return "root"
    if shutil.which("sudo") and _run(["sudo", "-n", "true"], timeout=5)[0] == 0:
        return "sudo"
    if shutil.which("pkexec") and (os.environ.get("DISPLAY") or
                                   os.environ.get("WAYLAND_DISPLAY")):
        return "pkexec"
    return None


def _alt_sizes(dev):
    """{altsetting: bytes per microframe} of the video-streaming interface
    (1), from the raw descriptors. sysfs's per-endpoint wMaxPacketSize drops
    the high-bandwidth multiplier (2 x 800 reads as 800), the descriptors
    keep it."""
    try:
        raw = (dev / "descriptors").read_bytes()
    except OSError:
        return {}
    sizes, i, intf, alt = {}, 0, None, None
    while i + 2 <= len(raw):
        ln, typ = raw[i], raw[i + 1]
        if ln < 2:
            break
        if typ == 4 and ln >= 4:                     # interface
            intf, alt = raw[i + 2], raw[i + 3]
        elif typ == 5 and ln >= 6 and intf == 1:     # endpoint
            if raw[i + 3] & 0x3 == 1:                # isochronous
                v = raw[i + 4] | (raw[i + 5] << 8)
                sizes[alt] = (v & 0x7FF) * (1 + ((v >> 11) & 3))
        i += ln
    return sizes


def reservations():
    """{bus: {"controller", "bytes", "cams": [...]}} for streaming cameras:
    each camera's active alternate setting and its packet size (sysfs and
    the descriptors in sysfs -- no root, cameras keep streaming)."""
    out = {}
    if not SYS_USB.exists():
        return out
    for dev in sorted(SYS_USB.iterdir()):
        name = dev.name
        if ":" in name or name.startswith("usb"):
            continue
        si = SYS_USB / f"{name}:1.1"
        if _read(si / "bInterfaceClass") != "0e":
            continue
        alt = int(_read(si / "bAlternateSetting") or 0)
        size = _alt_sizes(dev).get(alt, 0) if alt else 0
        bus = int(_read(dev / "busnum") or 0)
        b = out.setdefault(bus, {"controller": (SYS_USB / f"usb{bus}").resolve().parent.name,
                                 "bytes": 0, "cams": []})
        b["bytes"] += size
        b["cams"].append({"port": name, "product": _read(dev / "product"),
                          "alt": alt, "bytes": size})
    return out


class UvcModule:
    def __init__(self):
        self.job = {"state": "idle", "log": "", "kind": None}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ status
    def status(self):
        st = {"platform": sys.platform, "supported": sys.platform.startswith("linux")}
        if not st["supported"]:
            st["reason"] = ("The bandwidth patch is a Linux kernel driver change; "
                            "not applicable on this OS.")
            return st
        kver = platform.release()
        loaded = pathlib.Path("/sys/module/uvcvideo").exists()
        cap = _read(PARAMS / "mjpeg_max_payload")
        rc, path = _run(["modinfo", "-n", "uvcvideo"])
        installed_path = path if rc == 0 else None
        conf = _read(CONF)
        built_vm = _vermagic(BUILT_KO) if BUILT_KO.exists() else None
        hdr = pathlib.Path(f"/lib/modules/{kver}/build").exists()
        tools = {t: bool(shutil.which(t)) for t in ("git", "make", "gcc")}
        rc_sb, sb = _run(["mokutil", "--sb-state"])
        secure_boot = (None if rc_sb != 0 and "enabled" not in sb.lower()
                       else "enabled" in sb.lower())
        st.update({
            "kernel": kver,
            "loaded": loaded,
            "patched_loaded": bool(cap),
            "cap": int(cap) if cap.isdigit() else None,
            "quirks": _read(PARAMS / "quirks") or None,
            "installed_path": installed_path,
            "installed_patched": bool(installed_path and "/updates/" in installed_path),
            "conf": conf or None,
            "conf_has_cap": "mjpeg_max_payload" in conf,
            "built": BUILT_KO.exists(),
            "built_matches": built_vm == kver,
            "built_vermagic": built_vm,
            "prereqs": {"headers": hdr, **tools},
            "prereqs_ok": hdr and all(tools.values()),
            "install_hint": install_hint(kver),
            "secure_boot": secure_boot,
            "privilege": _privilege(),
            "default_cap": DEFAULT_CAP,
            "job": dict(self.job),
            "reservations": reservations(),
            "permanent": [
                f"sudo mkdir -p /lib/modules/{kver}/updates",
                f"sudo cp {BUILT_KO} /lib/modules/{kver}/updates/",
                "sudo depmod -a",
                f"echo 'options uvcvideo quirks=128 mjpeg_max_payload={DEFAULT_CAP}'"
                f" | sudo tee {CONF}",
            ],
            "undo": [
                f"sudo rm /lib/modules/{kver}/updates/uvcvideo.ko",
                "sudo depmod -a",
                f"remove mjpeg_max_payload=... from {CONF}",
            ],
        })
        return st

    # ------------------------------------------------------------- build
    def build(self):
        with self._lock:
            if self.job["state"] == "running":
                return {"ok": False, "error": "a job is already running"}
            self.job = {"state": "running", "log": "", "kind": "build"}
        threading.Thread(target=self._build, daemon=True).start()
        return {"ok": True}

    def _build(self):
        try:
            p = subprocess.Popen(["bash", str(EXTRA / "build.sh")], cwd=str(EXTRA),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True)
            for line in p.stdout:
                self.job["log"] += line
            rc = p.wait()
            self.job["state"] = "ok" if rc == 0 else "failed"
        except OSError as e:
            self.job["log"] += str(e)
            self.job["state"] = "failed"

    # -------------------------------------------------------- load/restore
    def _root_cmd(self, script):
        """Run a fixed root shell snippet with the best available method."""
        how = _privilege()
        if how == "root":
            cmd = ["sh", "-c", script]
        elif how == "sudo":
            cmd = ["sudo", "-n", "sh", "-c", script]
        elif how == "pkexec":
            cmd = ["pkexec", "sh", "-c", script]
        else:
            return {"ok": False, "manual": f"sudo sh -c '{script}'",
                    "error": "no way to get root from here — run the command "
                             "shown in a terminal (stop tracking first)"}
        rc, out = _run(cmd, timeout=120)
        if rc != 0:
            return {"ok": False, "error": out or f"exit {rc}",
                    "manual": f"sudo sh -c '{script}'"}
        return {"ok": True, "output": out, "via": how}

    def load(self, cap=DEFAULT_CAP):
        cap = int(cap)
        if not 256 <= cap <= 3072:
            return {"ok": False, "error": "cap must be 256-3072 bytes"}
        if not BUILT_KO.exists():
            return {"ok": False, "error": "build the patched driver first"}
        if _vermagic(BUILT_KO) != platform.release():
            return {"ok": False, "error": "the built driver is for a different "
                    "kernel — build it again"}
        quirks = _read(PARAMS / "quirks")
        q = f" quirks={quirks}" if quirks and quirks != "4294967295" else ""
        script = (f"modprobe uvcvideo && rmmod uvcvideo && "
                  f"insmod {BUILT_KO} mjpeg_max_payload={cap}{q}")
        return self._root_cmd(script)

    def restore(self):
        return self._root_cmd("rmmod uvcvideo && modprobe uvcvideo")


def wait_for_cameras(list_cameras, before, timeout=12.0):
    """After a driver swap the cameras re-enumerate; wait for them."""
    deadline = time.time() + timeout
    got = []
    while time.time() < deadline:
        got = list_cameras()
        if len(got) >= before:
            break
        time.sleep(0.5)
    return len(got)
