#!/bin/bash
# Run SafeEdge LIVE pipeline inside dustynv/l4t-pytorch with D455 + GPU YOLO.
# Usage: ./tools/docker_live.sh [--source realsense|webcam] [--mock]

set -e
PROJECT="/home/icp/safeedge/qwen_cloud"
IMAGE="dustynv/l4t-pytorch:r36.4.0"
SOURCE="${1:---source realsense}"

echo "=== SafeEdge LIVE in L4T container ==="
echo "Source: ${SOURCE}"
echo ""

# pyrealsense2 lives in the patient_monitor venv on the host — mount it in
REALSENSE_LIB="/home/icp/patient_monitor/patient_monitor_env/lib/python3.8/site-packages"

docker run --rm --runtime nvidia \
  --privileged \
  --network host \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${PROJECT}:/safeedge" \
  -v "${REALSENSE_LIB}:/host_rs_lib:ro" \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /dev:/dev \
  -e PYTHONPATH=/safeedge \
  "${IMAGE}" \
  bash -c "
    echo '--- Install deps (numpy<2 required) ---'
    pip install 'numpy<2' ultralytics supervision rtamt --index-url https://pypi.org/simple/ -q 2>&1 | tail -2

    echo '--- Try to find pyrealsense2 ---'
    python3 -c 'import sys; sys.path.insert(0, \"/host_rs_lib\"); import pyrealsense2; print(\"pyrealsense2:\", pyrealsense2.__version__)' 2>/dev/null || \
      pip install pyrealsense2 --index-url https://pypi.org/simple/ -q

    echo '--- Starting SafeEdge live pipeline ---'
    python3 /safeedge/edge/main.py ${SOURCE} \
      --model yolov8s.pt \
      --conf 0.25
  "
