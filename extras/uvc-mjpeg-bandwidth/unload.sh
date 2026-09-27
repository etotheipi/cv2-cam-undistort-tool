#!/usr/bin/env bash
# Put the stock uvcvideo driver back (same as rebooting).
# Stop anything using a camera first.
set -euo pipefail
sudo rmmod uvcvideo
sudo modprobe uvcvideo
if [[ -e /sys/module/uvcvideo/parameters/mjpeg_max_payload ]]; then
  echo "Still the patched module?! Check: modinfo -n uvcvideo" >&2
  exit 1
fi
echo "Stock uvcvideo loaded ($(modinfo -n uvcvideo))."
