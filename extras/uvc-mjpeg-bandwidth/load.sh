#!/usr/bin/env bash
# Temporarily swap the stock uvcvideo driver for the patched build.
#   ./load.sh [cap_bytes]        default cap 1600; 0 = behave exactly like stock
# Nothing is installed: a reboot, or ./unload.sh, restores the stock driver.
# Stop anything using a camera first (the bridge server, browsers, …).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
ko="$here/build/uvcvideo.ko"
cap="${1:-1600}"
quirks="$(cat /sys/module/uvcvideo/parameters/quirks 2>/dev/null || echo)"

[[ -f "$ko" ]] || { echo "No $ko — run ./build.sh first." >&2; exit 1; }
if users="$(fuser /dev/video* 2>/dev/null)" && [[ -n "${users// /}" ]]; then
  echo "Cameras are in use by PIDs:$users — stop them (e.g. the bridge server) first." >&2
  exit 1
fi

# modprobe first so the stock module's dependencies (videobuf2, …) are
# loaded; insmod doesn't resolve them itself
sudo modprobe uvcvideo
sudo rmmod uvcvideo
# keep a quirks value set via /etc/modprobe.d (insmod ignores that file)
args=("mjpeg_max_payload=$cap")
[[ -n "$quirks" && "$quirks" != "4294967295" ]] && args+=("quirks=$quirks")
sudo insmod "$ko" "${args[@]}"

p=/sys/module/uvcvideo/parameters
echo "Patched uvcvideo loaded: mjpeg_max_payload=$(cat $p/mjpeg_max_payload) quirks=$(cat $p/quirks)"
echo "Change the cap live (applies at the next stream start):"
echo "  echo 800 | sudo tee /sys/module/uvcvideo/parameters/mjpeg_max_payload"
