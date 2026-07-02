#!/bin/bash
# Run the FULL SafeEdge edge pipeline live inside dustynv/l4t-pytorch:
#   D455 → YOLOv8s(GPU) → ByteTrack → SignalExtractor → STL(ρ) → Intervention
#   → cloud_client → Alibaba Function Compute backend → dashboard
#
# Preview shows on VNC display :1. Events/state post to SAFEEDGE_CLOUD_URL.
# Usage: bash tools/docker_live.sh [--source realsense|webcam|mock]

set -e
PROJECT="/home/icp/safeedge/qwen_cloud"
IMAGE="dustynv/l4t-pytorch:r36.4.0"
SOURCE="${*:---source realsense}"
DISPLAY="${DISPLAY:-:1}"
CLOUD_URL="${SAFEEDGE_CLOUD_URL:-https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run}"

echo "=== SafeEdge FULL pipeline (edge → Alibaba cloud) ==="
echo "Source:    ${SOURCE}"
echo "Cloud URL: ${CLOUD_URL}"
echo "Display:   ${DISPLAY}"
echo ""

xhost +local:docker 2>/dev/null || true

docker run --rm --runtime nvidia \
  --privileged \
  --network host \
  -e DISPLAY=":1" \
  -e XAUTHORITY="/tmp/.Xauthority" \
  -e SAFEEDGE_CLOUD_URL="${CLOUD_URL}" \
  -e PYTHONPATH=/safeedge \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/icp/.Xauthority:/tmp/.Xauthority:ro \
  -v "${PROJECT}:/safeedge" \
  -v /dev/bus/usb:/dev/bus/usb \
  --device /dev/video0 --device /dev/video1 \
  -v /usr/lib/aarch64-linux-gnu/libusb-1.0.so.0:/usr/lib/aarch64-linux-gnu/libusb-1.0.so.0:ro \
  -v /usr/lib/aarch64-linux-gnu/libusb-1.0.so.0.3.0:/usr/lib/aarch64-linux-gnu/libusb-1.0.so.0.3.0:ro \
  "${IMAGE}" \
  bash -c "
    echo '--- Installing edge deps (numpy<2 required for torch 2.4) ---'
    pip install 'numpy<2' ultralytics supervision pyrealsense2 rtamt pyyaml python-dotenv \
      --index-url https://pypi.org/simple/ -q 2>&1 | tail -3

    echo '--- Checking D455 ---'
    python3 -c \"import pyrealsense2 as rs; print('RealSense devices:', rs.context().query_devices().size())\"

    echo '--- Starting full edge pipeline (Q in preview or Ctrl-C to stop) ---'
    cd /safeedge && python3 -m edge.main ${SOURCE} --model /safeedge/yolov8s.pt --conf 0.25
  "
