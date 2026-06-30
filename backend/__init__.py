"""SafeEdge cloud backend — FastAPI service deployed on Alibaba Cloud.

Receives safety events/state from the edge (Jetson), runs the three Qwen Cloud
skills (Policy Manager, Incident Reporter, Risk Forecaster), persists to
Tablestore, and serves the live dashboard.
"""
