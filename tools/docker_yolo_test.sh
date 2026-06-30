#!/bin/bash
# Test YOLO GPU inference inside NVIDIA L4T PyTorch container.
# Run from: /home/icp/safeedge/qwen_cloud

IMAGE="dustynv/l4t-pytorch:r36.4.0"
PROJECT="/home/icp/safeedge/qwen_cloud"

echo "=== YOLO GPU test in L4T container ==="
docker run --rm --runtime nvidia \
  -v "${PROJECT}:/safeedge" \
  -v "/home/icp/.config/Ultralytics:/root/.config/Ultralytics" \
  "${IMAGE}" \
  python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Memory:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')

# Install ultralytics if not present
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics', '-q'])

from ultralytics import YOLO
import time, cv2, numpy as np

model = YOLO('yolov8s.pt')
frame = np.zeros((480, 848, 3), dtype=np.uint8)

# Warmup
model.predict(frame, device=0, verbose=False)

# Benchmark 50 frames
t0 = time.perf_counter()
for _ in range(50):
    model.predict(frame, device=0, conf=0.35, verbose=False)
elapsed = time.perf_counter() - t0
fps = 50 / elapsed
print(f'GPU inference: {fps:.1f} fps  ({elapsed/50*1000:.1f}ms/frame)')
print('YOLO on GPU: OK' if fps > 10 else 'WARNING: low FPS')
"
