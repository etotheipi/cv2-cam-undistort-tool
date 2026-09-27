#!/usr/bin/env bash
# Build a patched uvcvideo.ko for the running kernel. Builds only; nothing
# is loaded or installed (see load.sh). Needs git, make, gcc and the
# kernel headers (`sudo apt install linux-headers-$(uname -r)`).
#
# Source: the mainline kernel tag matching the running major.minor
# (e.g. 7.0.0-28-generic -> v7.0), sparse-checked-out to just the uvc
# driver. Distro kernels may carry small uvc backports on top of that tag;
# fine for trying this out, but a permanent install should build from the
# distro's own kernel source instead.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
kver="$(uname -r)"
tag="v$(echo "$kver" | cut -d. -f1,2)"   # 7.0.0-28-generic -> v7.0, 6.10.x -> v6.10
kbuild="/lib/modules/$kver/build"
src="$here/build/linux-$tag"
uvc="$src/drivers/media/usb/uvc"

[[ -d "$kbuild" ]] || {
  echo "No kernel headers at $kbuild — install linux-headers-$kver first." >&2
  exit 1
}

if [[ ! -d "$src/.git" ]]; then
  echo "Fetching uvc driver source for $tag (sparse checkout, a few MB)…"
  mkdir -p "$here/build"
  git clone -q --depth 1 --filter=blob:none --sparse --branch "$tag" \
    https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git "$src"
  git -C "$src" sparse-checkout set drivers/media/usb/uvc
fi

# start from pristine source every time, then apply the patch
git -C "$src" checkout -q -- drivers/media/usb/uvc
git -C "$src" apply "$here/uvc-mjpeg-max-payload.patch"

make -C "$kbuild" M="$uvc" CONFIG_USB_VIDEO_CLASS=m clean >/dev/null
make -C "$kbuild" M="$uvc" CONFIG_USB_VIDEO_CLASS=m modules 2>&1 \
  | grep -vE '^  (CC|LD|MODPOST|BTF)|^make'
cp "$uvc/uvcvideo.ko" "$here/build/uvcvideo.ko"

echo
echo "Built $here/build/uvcvideo.ko for $kver:"
modinfo "$here/build/uvcvideo.ko" | grep -E '^vermagic|mjpeg_max_payload'
echo "Next: ./load.sh 1600   (temporary — a reboot restores the stock driver)"
