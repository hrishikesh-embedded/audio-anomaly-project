#!/usr/bin/env python3
import os
import time
import wave
import pyaudio
import numpy as np
import requests
from datetime import datetime
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============== Configuration ==============
BACKEND_URL = "http://10.94.70.136:8080/api/alerts"  # teammate backend
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEVICE_INDEX = 0  # change if needed
PRE_SECONDS = 5        # seconds of audio BEFORE anomaly
POST_SECONDS = 3       # seconds of audio AFTER anomaly
AUDIO_SAVE_DIR = "/var/www/html/audio-storage"
SENSOR_ID = "TI-Board-001"

os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)


class ConservativeAudioSystem:
    def __init__(self):
        # ML components
        self.model = IsolationForest(contamination=0.01, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Audio params
        self.CHUNK = CHUNK
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.DEVICE_INDEX = DEVICE_INDEX
        self.RATE = self.find_sample_rate()

        # Pre-anomaly rolling buffer
        self.pre_buffer_len = int((self.RATE * PRE_SECONDS) // self.CHUNK)
        self.pre_audio_buffer = deque(maxlen=self.pre_buffer_len)

    def find_sample_rate(self):
        """Probe common sample rates and return the first that opens successfully."""
        rates = [48000, 16000, 44100, 8000, 22050]
        p = pyaudio.PyAudio()
        for rate in rates:
            try:
                stream = p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=rate,
                    input=True,
                    input_device_index=self.DEVICE_INDEX,
                    frames_per_buffer=self.CHUNK,
                )
                stream.close()
                p.terminate()
                return rate
            except Exception:
                continue
        p.terminate()
        return 48000  # fallback

    def extract_features(self, audio_data: np.ndarray):
        """Compute lightweight features for anomaly detection."""
        # Convert int16 to float [-1,1]
        audio = audio_data.astype(np.float32) / 32768.0
        # Time-domain features
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        zero_crossings = int(np.sum(np.diff(np.sign(audio)) != 0))
        # Spectral centroid
        spectrum = np.fft.rfft(audio)
        mag = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.RATE)
        spectral_centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
        # Crest factor
        crest_factor = float(peak / rms) if rms > 0 else 0.0
        return [rms, peak, zero_crossings, spectral_centroid, crest_factor]

    def save_audio_clip(self, audio_clip: np.ndarray, timestamp_iso: str) -> str:
        """Save concatenated int16 audio to WAV and return the file path."""
        safe_ts = timestamp_iso.replace(":", "-")
        filename = f"anomaly_{safe_ts}.wav"
        filepath = os.path.join(AUDIO_SAVE_DIR, filename)
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(self.RATE)
            wav_file.writeframes(audio_clip.tobytes())
        return filepath

    def send_alert_with_file(self, filepath: str, anomaly_type: str = "Anomaly"):
        """POST the WAV file with metadata to the backend."""
        try:
            with open(filepath, 'rb') as f:
                files = {'audio_file': (os.path.basename(filepath), f, 'audio/wav')}
                data = {
                    'sensorId': SENSOR_ID,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'anomalyType': anomaly_type,
                }
                resp = requests.post(BACKEND_URL, files=files, data=data, timeout=10)
                print(f"[ALERT] POST {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[ALERT] Error sending alert: {e}")

    def calibrate_with_real_audio(self, duration: int = 30) -> bool:
        """Record ambient audio for 'duration' seconds and train the model."""
        print(f"[CALIBRATE] Starting {duration}s ambient capture at {self.RATE} Hz...")
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.DEVICE_INDEX,
                frames_per_buffer=self.CHUNK,
            )
        except Exception as e:
            print(f"[CALIBRATE] Failed to open stream: {e}")
            p.terminate()
            return False

        X = []
        frames_needed = int((self.RATE * duration) // self.CHUNK)
        try:
            for _ in range(frames_needed):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                feats = self.extract_features(audio)
                X.append(feats)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if len(X) < 10:
            print("[CALIBRATE] Not enough data collected.")
            return False

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        print("[CALIBRATE] Completed. Model trained on ambient audio.")
        return True

    def start_monitoring(self):
        """Continuous monitoring loop with pre/post buffering and alerting."""
        if not self.is_trained:
            print("[MONITOR] Model not trained. Run calibration first.")
            return

        print("[MONITOR] Starting continuous monitoring. Press Ctrl+C to stop.")
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.DEVICE_INDEX,
                frames_per_buffer=self.CHUNK,
            )
        except Exception as e:
            print(f"[MONITOR] Failed to open stream: {e}")
            p.terminate()
            return

        try:
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)

                # Maintain pre-event buffer
                self.pre_audio_buffer.append(audio)

                # Predict
                feats = np.asarray(self.extract_features(audio), dtype=np.float32).reshape(1, -1)
                feats_scaled = self.scaler.transform(feats)
                pred = self.model.predict(feats_scaled)  # -1 anomaly, 1 normal

                # Simple status print
                rms = np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2))
                level = int(min(20, rms * 200))
                bar = "#" * level
                status = "ANOMALY" if pred == -1 else "NORMAL "
                print(f"[{status}] Level:{bar:<20}", end="\r")

                if pred == -1:
                    print()  # newline after anomaly line
                    ts = datetime.utcnow().isoformat() + "Z"
                    print(f"[EVENT] Anomaly at {ts}. Capturing post audio...")

                    # Capture post-event audio
                    post_chunks = int((self.RATE * POST_SECONDS) // self.CHUNK)
                    post_audio = []
                    for _ in range(post_chunks):
                        d2 = stream.read(self.CHUNK, exception_on_overflow=False)
                        post_audio.append(np.frombuffer(d2, dtype=np.int16))

                    # Concatenate pre + post
                    pre_list = list(self.pre_audio_buffer)
                    full_clip = np.concatenate(pre_list + post_audio, axis=0)

                    # Save and alert
                    filepath = self.save_audio_clip(full_clip, ts)
                    print(f"[EVENT] Saved: {filepath}")
                    self.send_alert_with_file(filepath, anomaly_type="AudioAnomaly")

                    # Optional cooldown to avoid rapid duplicate alerts
                    time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[MONITOR] Stopped by user.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    print("=" * 60)
    system = ConservativeAudioSystem()
    if system.calibrate_with_real_audio(duration=30):
        system.start_monitoring()
    else:
        print("[MAIN] Calibration failed. Exiting.")


if __name__ == "__main__":
    main()
