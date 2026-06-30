#!/bin/bash
# Launch SafeEdge live recorder inside dustynv L4T container.
# D455 via USB passthrough, GPU YOLO, annotated MP4 saved with timestamps.
# Usage: bash tools/docker_live_record.sh

set -e
PROJECT="/home/icp/safeedge/qwen_cloud"
IMAGE="dustynv/l4t-pytorch:r36.4.0"
DISPLAY="${DISPLAY:-:1}"

echo "=== SafeEdge Live Recorder ==="
echo "Image:   ${IMAGE}"
echo "Display: ${DISPLAY}"
echo ""

# Allow docker to use the X display
xhost +local:docker 2>/dev/null || true

docker run --rm --runtime nvidia \
  --privileged \
  --network host \
  -e DISPLAY=":1" \
  -e XAUTHORITY="/tmp/.Xauthority" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/icp/.Xauthority:/tmp/.Xauthority:ro \
  -v "${PROJECT}:/safeedge" \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /usr/lib/aarch64-linux-gnu/libusb-1.0.so.0:/usr/lib/aarch64-linux-gnu/libusb-1.0.so.0:ro \
  -v /usr/lib/aarch64-linux-gnu/libusb-1.0.so.0.3.0:/usr/lib/aarch64-linux-gnu/libusb-1.0.so.0.3.0:ro \
  "${IMAGE}" \
  bash -c "
    echo '--- Installing Python deps ---'
    pip install 'numpy<2' ultralytics pyrealsense2 \
      --index-url https://pypi.org/simple/ -q 2>&1 | tail -2

    echo '--- Checking D455 ---'
    python3 -c \"
import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
print(f'RealSense devices: {devs.size()}')
for i in range(devs.size()):
    print(' ', devs[i].get_info(rs.camera_info.name))
\"

    echo '--- Starting live recorder (Q to stop) ---'
    python3 /safeedge/tools/live_record.py --out-dir /safeedge/data/captures
  "
