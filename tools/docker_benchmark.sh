#!/bin/bash
# Run SafeEdge offline benchmark inside dustynv/l4t-pytorch (Jetson GPU)
# Usage: ./tools/docker_benchmark.sh [max_frames]

set -e
PROJECT="/home/icp/safeedge/qwen_cloud"
IMAGE="dustynv/l4t-pytorch:r36.4.0"
MAX_FRAMES="${1:-450}"

echo "=== SafeEdge GPU Benchmark in L4T container ==="
echo "Image:  ${IMAGE}"
echo "Frames: ${MAX_FRAMES}"
echo ""

docker run --rm --runtime nvidia \
  -v "${PROJECT}:/safeedge" \
  -e PYTHONPATH=/safeedge \
  "${IMAGE}" \
  bash -c "
    set -e
    echo '--- Install ultralytics into container (numpy<2 required for torch 2.4) ---'
    pip install 'numpy<2' ultralytics supervision --index-url https://pypi.org/simple/ -q

    echo '--- CUDA check ---'
    python3 -c \"
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,2), 'GB')
\"

    echo '--- Run benchmark on GPU ---'
    python3 /safeedge/tools/benchmark_detector.py \
      --video /safeedge/data/captures/carpark_daylight_01.mp4 \
      --models yolov8n.pt yolov8s.pt yolov8m.pt \
      --confs 0.25 0.35 0.45 \
      --device 0 \
      --max-frames ${MAX_FRAMES} \
      --no-video
  "
