"""
Pengukuran real-time DI RASPBERRY PI 5 (platform=edge) — menjawab REQ-2.3
(breakdown latensi on-device: detector / preprocess / inference / network-send),
REQ-2.4 (latensi inferensi model TFLite sebenarnya di Pi, untuk memverifikasi klaim
~13×), dan menyuplai data untuk REQ-2.1/3.1 (sesi berulang -> CI via
aggregate_sessions.py).

Skrip ini MENIRU pipeline app2_rpi.py PERSIS (rpicam-vid MJPEG -> MediaPipe
FaceDetection -> ROI mata/mulut -> TFLite MobileNetV2), tetapi menambahkan:
  * timing per-tahap: capture, face_detect, preprocess, eye_infer, mouth_infer;
  * benchmark network-send terpisah (--bench_network N): rata-rata latensi POST
    event ke endpoint (.env EVENT_ENDPOINT), karena alert asli jarang;
  * log per-frame opsional (--session_log): t_rel_s,eye_label,eye_conf,
    mouth_label,mouth_conf,face_detected -> dipakai score_annotated_session.py
    (accuracy/false-alarm dgn GT, dan sweep ambang REQ-1.6);
  * satu baris ringkas di-APPEND ke realtime_eval/realtime_baseline_results.csv
    dengan platform="edge" sehingga langsung kompatibel dengan
    aggregate_sessions.py (kolom sama seperti versi desktop).

Jalankan DI PI (setelah menyalin model .tflite + app2_rpi.py, lih.
journal/RASPBERRY_PI_DATA_COLLECTION.md):

  python3 realtime_eval/measure_realtime_pi.py --condition normal_light_frontal \
        --duration 30 --session_log sessions/p01_normal.csv
  python3 realtime_eval/measure_realtime_pi.py --bench_network 30 --condition net_bench

CATATAN: skrip ini butuh hardware Pi (rpicam-vid), MediaPipe, dan runtime TFLite —
tidak bisa dijalankan di mesin dev tanpa kamera Pi. Semua dependensi di-import
secara defensif dengan pesan yang jelas bila hilang.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from urllib import error, request

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("[ERROR] opencv belum terinstal: pip install opencv-python")

# Runtime TFLite: sama seperti app2_rpi.py (tflite_runtime -> ai_edge_litert -> tf.lite)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("[ERROR] MediaPipe belum terinstal: pip install mediapipe")

try:
    import psutil
except ImportError:
    psutil = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
IMG_SIZE = 160
MIN_CONF = 0.75


class RpicamCapture:
    """Identik dengan _RpicamCapture di app2_rpi.py (rpicam-vid MJPEG pipe)."""
    def __init__(self, width=640, height=480, framerate=15):
        subprocess.run(["pkill", "-f", "rpicam-vid"], capture_output=True)
        time.sleep(0.2)
        cmd = ["rpicam-vid", "-t", "0", "--codec", "mjpeg", "--nopreview",
               "--width", str(width), "--height", str(height),
               "--framerate", str(framerate), "--mode", "1536:864:10:P", "-o", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._buf = b""

    def read(self):
        while True:
            if self._proc.poll() is not None:
                return False, None
            chunk = self._proc.stdout.read(8192)
            if not chunk:
                return False, None
            self._buf += chunk
            s = self._buf.find(b"\xff\xd8"); e = self._buf.find(b"\xff\xd9")
            if s != -1 and e != -1 and e > s:
                jpg = self._buf[s:e + 2]; self._buf = self._buf[e + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame

    def release(self):
        self._proc.terminate(); self._proc.wait()


def load_tflite(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def predict(img, interp, ins, outs, labels, safe_index):
    if img is None or img.size == 0:
        return labels[safe_index], 0.0
    x = np.expand_dims(cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0, 0)
    interp.set_tensor(ins[0]["index"], x)
    interp.invoke()
    pred = interp.get_tensor(outs[0]["index"])[0]
    idx = int(np.argmax(pred)); conf = float(pred[idx])
    return (labels[safe_index] if conf < MIN_CONF else labels[idx]), conf


def load_endpoint():
    """Endpoint event = sama seperti app2_rpi.py (DROWSINESS_API_URL di .env)."""
    url = os.environ.get("DROWSINESS_API_URL") or os.environ.get("EVENT_ENDPOINT")
    env = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line.startswith("DROWSINESS_API_URL=") or line.startswith("EVENT_ENDPOINT="):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
    return url


def bench_network(url, n):
    if not url:
        print("[network] EVENT_ENDPOINT tidak diset (.env) — lewati benchmark network.")
        return None
    import json
    lat = []
    payload = json.dumps({"event_type": "benchmark", "ts": time.time()}).encode()
    for _ in range(n):
        req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            request.urlopen(req, timeout=5).read()
        except error.URLError as e:
            print(f"[network] gagal POST: {e}"); continue
        lat.append((time.perf_counter() - t0) * 1000)
    if lat:
        print(f"[network] {len(lat)} POST -> avg {np.mean(lat):.1f} ms, p95 {np.percentile(lat,95):.1f} ms")
    return (float(np.mean(lat)), float(np.percentile(lat, 95))) if lat else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--session_log", default=None, help="CSV per-frame untuk score_annotated_session.py")
    ap.add_argument("--bench_network", type=int, default=0, help="jumlah POST untuk mengukur latensi network-send")
    ap.add_argument("--out_csv", default=os.path.join(THIS_DIR, "realtime_baseline_results.csv"))
    ap.add_argument("--eye_model", default=os.path.join(PROJECT_ROOT, "model_eye_mobilenet.tflite"))
    ap.add_argument("--mouth_model", default=os.path.join(PROJECT_ROOT, "model_mouth_mobilenet.tflite"))
    args = ap.parse_args()

    net = bench_network(load_endpoint(), args.bench_network) if args.bench_network else None

    cap = RpicamCapture()
    interp_e, ie, oe = load_tflite(args.eye_model)
    interp_m, im, om = load_tflite(args.mouth_model)
    face_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    stage = defaultdict(list)
    frame_lat, n_faces = [], 0
    cpu, ram = [], []
    log_rows = []
    proc = psutil.Process() if psutil else None
    t_start = time.time(); n_frames = 0

    print(f"[measure] kondisi={args.condition}, durasi={args.duration}s ...", flush=True)
    while time.time() - t_start < args.duration:
        tf0 = time.perf_counter()
        t0 = time.perf_counter(); ret, frame = cap.read(); t_cap = time.perf_counter() - t0
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        t0 = time.perf_counter()
        res = face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); t_face = time.perf_counter() - t0

        eye_label, eye_conf, mouth_label, mouth_conf, face_found = "Open", 0.0, "No_yawn", 0.0, 0
        if res.detections:
            face_found = 1; n_faces += 1
            d = res.detections[0].location_data.relative_bounding_box
            fh, fw, _ = frame.shape
            fx, fy = max(0, int(d.xmin * fw)), max(0, int(d.ymin * fh))
            fbw, fbh = min(fw - fx, int(d.width * fw)), min(fh - fy, int(d.height * fh))
            if fbw >= 50 and fbh >= 50:
                eye_roi = frame[fy:fy + int(fbh * 0.5), fx:fx + fbw]
                mx, my = fx + int(fbw * 0.2), fy + int(fbh * 0.7)
                mouth_roi = frame[my:my + int(fbh * 0.25), mx:mx + int(fbw * 0.6)]
                t0 = time.perf_counter()
                eye_label, eye_conf = predict(eye_roi, interp_e, ie, oe, ["Closed", "Open"], 1)
                t_eye = time.perf_counter() - t0
                t0 = time.perf_counter()
                mouth_label, mouth_conf = predict(mouth_roi, interp_m, im, om, ["No_yawn", "Yawn"], 0)
                t_mouth = time.perf_counter() - t0
                stage["eye_infer"].append(t_eye * 1000)
                stage["mouth_infer"].append(t_mouth * 1000)
        stage["capture"].append(t_cap * 1000)
        stage["face_detect"].append(t_face * 1000)
        frame_lat.append((time.perf_counter() - tf0) * 1000)
        n_frames += 1
        if proc:
            cpu.append(proc.cpu_percent()); ram.append(proc.memory_info().rss / 1e6)
        if args.session_log is not None:
            log_rows.append(dict(t_rel_s=round(time.time() - t_start, 3),
                                 face_detected=face_found, eye_label=eye_label,
                                 eye_conf=round(eye_conf, 4), mouth_label=mouth_label,
                                 mouth_conf=round(mouth_conf, 4)))
    cap.release()

    dur = time.time() - t_start
    fps = n_frames / dur if dur else 0.0
    def m(k): return round(float(np.mean(stage[k])), 2) if stage[k] else 0.0
    row = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        condition=args.condition, platform="edge",
        duration_s=round(dur, 1), n_frames=n_frames, fps_overall=round(fps, 2),
        avg_frame_latency_ms=round(float(np.mean(frame_lat)), 2) if frame_lat else 0.0,
        p95_frame_latency_ms=round(float(np.percentile(frame_lat, 95)), 2) if frame_lat else 0.0,
        avg_face_detect_ms=m("face_detect"), avg_eye_infer_ms=m("eye_infer"),
        avg_mouth_infer_ms=m("mouth_infer"),
        face_detect_rate_pct=round(100 * n_faces / n_frames, 1) if n_frames else 0.0,
        avg_cpu_pct=round(float(np.mean(cpu)), 1) if cpu else "",
        avg_ram_MB=round(float(np.mean(ram)), 1) if ram else "",
        network_send_avg_ms=round(net[0], 2) if net else "",
        network_send_p95_ms=round(net[1], 2) if net else "",
        accuracy_pct="", false_alert_rate_pct="",
    )
    # kolom disamakan namanya dgn measure_realtime_baseline.py agar aggregate cocok
    row["face_detect_rate_%"] = row.pop("face_detect_rate_pct")
    row["avg_cpu_%"] = row.pop("avg_cpu_pct")
    row["accuracy_%"] = row.pop("accuracy_pct")
    row["false_alert_rate_%"] = row.pop("false_alert_rate_pct")

    print("\n=== RINGKASAN (Pi edge) ===")
    for k, v in row.items():
        print(f"  {k:24s}: {v}")

    exists = os.path.exists(args.out_csv)
    # Union header supaya baris edge (punya kolom network_send_*) tetap konsisten dgn file lama.
    fieldnames = list(row.keys())
    if exists:
        with open(args.out_csv) as f:
            old = f.readline().strip().split(",")
        for c in old:
            if c not in fieldnames:
                fieldnames.append(c)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"\nBaris ringkas ditambahkan ke: {args.out_csv}")

    if args.session_log is not None and log_rows:
        os.makedirs(os.path.dirname(args.session_log) or ".", exist_ok=True)
        import pandas as pd
        pd.DataFrame(log_rows).to_csv(args.session_log, index=False)
        print(f"Log per-frame ({len(log_rows)} baris) -> {args.session_log} "
              f"(untuk score_annotated_session.py)")


if __name__ == "__main__":
    main()
