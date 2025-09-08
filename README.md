# Audio Anomaly Project

Building real‑time audio anomaly detection for edge devices. Calibrates on ambient audio, extracts RMS/peak/zero‑crossings/spectral‑centroid/crest‑factor features, detects with IsolationForest, buffers 5s pre + 3s post, saves WAV, and posts alerts to a backend.

- Tech: Python • NumPy • scikit‑learn (IsolationForest) • PyAudio • requests • deque/wave • Linux
- Design: conservative thresholds via calibration • dynamic sample‑rate probing • evidence‑first alerts with contextual clips
- Links: [Demo clip](#) • [Alert screenshot](#) • [Repository](#)

## What I built
- Real‑time audio anomaly detection with normalized features (RMS, peak, zero‑crossings, spectral centroid, crest factor) and IsolationForest
- Rolling context: 5s pre‑event buffer + 3s post‑event capture; saves timestamped WAV for review
- Robustness: environment calibration to reduce false positives; auto sample‑rate selection for different mics
- Delivery: multipart HTTP alerts with sensor ID, timestamps, and attached audio
