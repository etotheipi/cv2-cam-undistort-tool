#!/usr/bin/env python3
"""Live USB 2 isochronous reservation of every streaming UVC camera, per bus.

A camera reserves its bandwidth when it starts streaming, by selecting an
alternate setting of its video-streaming interface; each alternate setting
has a fixed max packet size. This reads the active setting from sysfs and
the packet sizes from `lsusb -v` — no root needed, cameras keep streaming.

Rule of thumb for one high-speed (480 Mbps) bus, from USB 2.0 §5.11.3:
periodic transfers may use 100 µs of each 125 µs microframe, and the
controller budgets worst-case bit stuffing, so ~4,800-5,000 bytes per
microframe in total. Strict controllers (e.g. Renesas uPD720202) refuse
anything above that; some (e.g. AMD chipsets) accept more.
"""
import collections
import pathlib
import re
import subprocess

SYS = pathlib.Path("/sys/bus/usb/devices")


def read(p):
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def packet_sizes(dev):
    """{altsetting: bytes per microframe} for interface 1 (video streaming)."""
    txt = subprocess.run(
        ["lsusb", "-v", "-s", f"{int(read(dev / 'busnum'))}:{int(read(dev / 'devnum'))}"],
        capture_output=True, text=True).stdout
    sizes, intf, alt = {}, None, None
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "bInterfaceNumber":
            intf = int(f[1])
        elif f[0] == "bAlternateSetting":
            alt = int(f[1])
        elif f[0] == "wMaxPacketSize" and intf == 1:
            m = re.search(r"(\d)x (\d+) bytes", line)
            if m:
                sizes[alt] = int(m[1]) * int(m[2])
    return sizes


def main():
    buses = collections.defaultdict(list)
    for dev in sorted(SYS.iterdir()):
        name = dev.name
        if ":" in name or name.startswith("usb"):
            continue
        stream_if = SYS / f"{name}:1.1"
        if read(stream_if / "bInterfaceClass") != "0e":
            continue                       # not a UVC camera
        alt = int(read(stream_if / "bAlternateSetting") or 0)
        size = packet_sizes(dev).get(alt, 0) if alt else 0
        buses[int(read(dev / "busnum"))].append((name, read(dev / "product"), alt, size))
    if not buses:
        print("No UVC cameras found.")
        return
    cap = read(pathlib.Path("/sys/module/uvcvideo/parameters/mjpeg_max_payload"))
    print("uvcvideo mjpeg_max_payload:", cap if cap else "(stock driver)")
    for bus, cams in sorted(buses.items()):
        ctrl = (SYS / f"usb{bus}").resolve().parent.name
        print(f"bus {bus} ({ctrl}): {sum(c[3] for c in cams)} B per 125 µs reserved")
        for name, product, alt, size in cams:
            state = f"alt {alt:2d} = {size:4d} B" if alt else "idle"
            print(f"   {name:8s} {product[:32]:32s} {state}")


if __name__ == "__main__":
    main()
