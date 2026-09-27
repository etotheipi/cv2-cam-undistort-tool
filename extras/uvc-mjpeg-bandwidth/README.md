# Extra: run more USB webcams per controller (uvcvideo MJPEG bandwidth cap)

**Optional, and not needed for the calibration tool itself.** This is a
23-line patch to Linux's webcam driver (`uvcvideo`), plus scripts to build it
and try it temporarily. Use it only if you run several USB 2 webcams on one
USB controller and some of them fail to start. Whether to adopt it is your
call; nothing here installs itself.

## The problem

Symptoms: a camera opens but never delivers frames, or a tile sits on
"connecting…". Meanwhile the kernel log shows:

```
usb 13-1.4: Not enough bandwidth for new device state.
usb 13-1.4: Not enough bandwidth for altsetting 6
```

(`journalctl -k | grep -i bandwidth`)

USB 2 webcams stream *isochronously*: when a stream starts, the camera
reserves a fixed slice of every 125 µs USB "microframe". The slice is not the
data it sends. It is a worst-case number from the camera's firmware, and the
driver trusts it for MJPEG. A USB 3 port doesn't help, because USB 2 cameras
only use the controller's 480 Mbps USB 2 side. Every camera on that
controller (including through hubs) shares one budget. By the USB 2.0 spec
(§5.11.3), that budget is 100 µs per microframe including per-packet overhead
and worst-case bit stuffing, which works out to roughly **4,800–5,000 bytes of
reservations per bus**. Strict controllers enforce exactly that. Others
accept more.

Measured on real hardware (each camera alone; "reserved" is what the
firmware requests, "actual" is the real MJPEG data rate):

| Camera | Mode | Reserved (B per 125 µs) | Actual (B per 125 µs) |
|---|---|---|---|
| Arducam OV9782 (`0c45:6366`) | 640×480 | 2,400 | ~50 |
| Arducam OV9782 | 800×600 – 1280×800 | **3,072** | ~175–195 |
| Innomaker U20CAM-1080p (`0c45:6366`) | 640×480 – 1080p | 2,400 | ~350–520 |
| Innomaker U20CAM-1080p | 1280×960 / 1024×576 | 1,600 / 800 | |
| Arducam 5MP (`1bcf:284c`) | any MJPEG mode | **3,060** | ~900 at 720p/1080p |

Controller behaviour seen with those numbers:

| Controller | Accepts | Refuses |
|---|---|---|
| Renesas uPD720202 (StarTech 4-port / 4-controller PCIe card) | 2 × 2,400 = 4,800 | 3,072 + 2,400; 3,060 + 2,400 |
| AMD Matisse / 500-series chipset (motherboard) | up to ~5,460 (beyond spec) | varies; often at the edge |

So the stock driver gets **one camera per controller at 720p** on a strict
controller, even though two cameras actually send under 10% of one
reservation.

`options uvcvideo quirks=128` (`UVC_QUIRK_FIX_BANDWIDTH`), commonly suggested
online, does **not** help here: the driver applies it to uncompressed formats
(YUYV) only, and these cameras stream MJPEG.

## What the patch does

`uvc_video_start_transfer()` picks the smallest alternate setting whose packet
size is at least the camera's claimed payload. The patch adds a module
parameter, `mjpeg_max_payload` (bytes). When it's set, the claimed payload of
**compressed** streams is capped at that value before the choice is made. The
camera then paces its frames into the smaller packets. See
[`uvc-mjpeg-max-payload.patch`](uvc-mjpeg-max-payload.patch) for the whole
change (3 files, +23 lines, nothing removed).

With `mjpeg_max_payload=1600`, all six cameras above ran **at 1280×720
simultaneously on the Renesas card: three on one port, two on another, one
on a third**. Each held ~30 fps, 1,800 of 1,800 frames arrived as complete
JPEGs, and the kernel logged no bandwidth errors. On the stock driver, that
card refused even two of these cameras at 720p on one port.

### Scope and risk

- **Only `uvcvideo` changes.** Keyboards, mice, storage, Bluetooth and camera
  microphones (`snd-usb-audio`) use other drivers.
- **`mjpeg_max_payload=0` (the default) is byte-for-byte stock behaviour.**
  Uncompressed (YUYV) streams are never affected.
- **The bus stays within spec.** The controller still budgets correctly for
  the smaller slice, and the camera cannot exceed it. This is safer than
  controllers that simply accept reservations beyond the spec.
- **Camera firmware must cope with smaller packets.** It advertises those
  alternate settings, so it should, but verify with `stream_test.py`.
  Mainline `uvcvideo` defaults to `nodrop=1`, so incomplete frames would be
  *delivered*, flagged as errors, rather than dropped. The test catches that.
- **Latency:** a smaller slice spreads each frame over more microframes. At
  1,600 B/125 µs, a 150 KB JPEG takes ~12 ms, well within the 33 ms frame
  period at 30 fps. Much smaller caps (800) get tight for large JPEGs.
- **Source version:** `build.sh` builds from the **mainline** tag matching your
  kernel (e.g. `v7.0` for `7.0.0-28-generic`). Distro kernels can carry small
  uvc backports. That's fine for trying it out, but for a permanent install,
  prefer building the same patch against your distro's own kernel source.

## Try it (temporary)

Requirements: `git`, `make`, `gcc`, `v4l-utils`, and kernel headers
(`sudo apt install linux-headers-$(uname -r)`). Secure Boot must be off, or
you must sign the module yourself.

```bash
cd extras/uvc-mjpeg-bandwidth
./build.sh              # fetches the uvc driver source, applies the patch, builds build/uvcvideo.ko
# stop anything using a camera (the bridge server, browser tabs), then:
./load.sh 1600          # swaps in the patched driver until reboot (asks for sudo)
python3 usb_reservations.py              # what each streaming camera has reserved, per bus
python3 stream_test.py 1280x720 11-1.1 11-1.2 11-1.3   # stress several cameras at once
./unload.sh             # back to the stock driver (or just reboot)
```

`load.sh` keeps whatever `quirks` value is active. The cap can be changed
while the module is loaded, and takes effect at each camera's next stream
start:

```bash
echo 800 | sudo tee /sys/module/uvcvideo/parameters/mjpeg_max_payload
```

### Choosing a cap

Real traffic has to fit, with headroom for busy scenes (bigger JPEGs). The
controller then decides how many caps fit per bus (~4,800 B on a strict
controller):

| Cap | Packet | Per strict bus | Fits |
|---|---|---|---|
| 1600 | 2×800 | 3 cameras | OV9782 (~9× headroom at 720p), Innomaker, 5MP (~1.8×) |
| 800 | 1×800 | ~6 cameras | OV9782 only (~4.6×); too tight for the 5MP |

## Making it permanent (your decision, untested here)

Not done on the reference machine. A straightforward manual route:

```bash
sudo mkdir -p /lib/modules/$(uname -r)/updates
sudo cp build/uvcvideo.ko /lib/modules/$(uname -r)/updates/
sudo depmod -a
echo 'options uvcvideo quirks=128 mjpeg_max_payload=1600' | sudo tee /etc/modprobe.d/uvcvideo.conf
```

`updates/` takes precedence over the stock module. It is **per kernel
version**, so after every kernel update the stock driver comes back until you
rebuild and copy again. Packaging the patch with DKMS automates that. To
undo, delete the copied file, run `sudo depmod -a`, and remove
`mjpeg_max_payload=…` from the options line.

## Other ways around the limit

- **One camera per controller.** More controllers (e.g. a second
  multi-controller PCIe card) or motherboard/GPU USB ports.
- **Lower-bandwidth modes.** The OV9782s reserve only 2,400 at 640×480, which
  lets two share a strict controller. Some firmware reserves less at specific
  modes (e.g. the Innomaker at 1280×960).
- **Dim-light frame rate is a separate issue.** Many cameras halve their
  frame rate in low light when `exposure_dynamic_framerate=1`. That's
  exposure, not bandwidth:
  `v4l2-ctl -d /dev/videoN -c exposure_dynamic_framerate=0`.
