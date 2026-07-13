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

Skrip ini juga SUDAH MENGGABUNG pemandu sesi (dulu session_cue.py): dengan
--scripted ia menampilkan aba-aba drowsiness beranotasi TEPAT WAKTU di layar
(otomatis SINKRON t=0 karena satu proses dengan recorder), lalu di AKHIR sesi
OTOMATIS menskor log per-frame terhadap ground-truth bawaan (EVENTS) memakai
engine dari score_annotated_session.py — sehingga kolom accuracy_% dan
false_alert_rate_% yang dulu KOSONG kini terisi tanpa langkah manual terpisah.

Jalankan DI PI (setelah menyalin model .tflite + app2_rpi.py, lih.
journal/RASPBERRY_PI_DATA_COLLECTION.md):

  # panduan "mau ambil kondisi apa" (latensi vs akurasi) + contoh perintah:
  python3 realtime_eval/measure_realtime_pi.py --list_conditions
  # A. LATENSI/FPS (tanpa partisipan, tanpa --scripted) — accuracy_% dibiarkan kosong:
  python3 realtime_eval/measure_realtime_pi.py --condition lat_pi_picam --duration 30
  # B. AKURASI (partisipan, aba-aba live + AUTO-ISI accuracy_%/false_alert_rate_%):
  python3 realtime_eval/measure_realtime_pi.py --scripted \
        --condition p01_picam_60_150 --session_log sessions/p01_picam_60_150.csv
  # benchmark network-send terpisah:
  python3 realtime_eval/measure_realtime_pi.py --bench_network 30 --condition lat_net
  # lihat kartu jadwal / tulis ground-truth CSV (untuk skoring manual):
  python3 realtime_eval/measure_realtime_pi.py --card_only
  python3 realtime_eval/measure_realtime_pi.py --write_truth sessions/truth_scripted.csv

CATATAN: skrip ini butuh hardware Pi (rpicam-vid), MediaPipe, dan runtime TFLite —
tidak bisa dijalankan di mesin dev tanpa kamera Pi. Semua dependensi di-import
secara defensif dengan pesan yang jelas bila hilang.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
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

# ===========================================================================
#  PEMANDU SESI TERSCRIPT (digabung dari session_cue.py)
#  Mode --scripted menampilkan aba-aba drowsiness beranotasi TEPAT WAKTU selama
#  perekaman (otomatis SINKRON t=0 karena satu proses dengan recorder), lalu di
#  akhir sesi OTOMATIS menskor log per-frame vs ground-truth (EVENTS) sehingga
#  kolom accuracy_% & false_alert_rate_% terisi (tidak lagi kosong).
#
#  PENTING soal kedip: kedip NORMAL (<0,4 dtk) TIDAK memicu apa-apa (ada toleransi
#  kedip). Yang memicu microsleep/critical = mata TERTUTUP BERKELANJUTAN, bukan
#  kedip. Saat fase "NORMAL", kedip sewajarnya.
# ===========================================================================

# (detik_mulai, instruksi). Aba-aba berlaku sampai entri berikutnya.
SCHEDULE = [
    (0,   "NORMAL — hadap kamera, KEDIP BIASA (aman, tidak ada alert)"),
    (30,  "TUTUP MATA TERUS — jangan dibuka  (memicu MICROSLEEP di detik ~45)"),
    (50,  "NORMAL — buka mata, kedip biasa"),
    (70,  "MENGUAP #1 — buka mulut lebar ~1 dtk"),
    (75,  "NORMAL sebentar (mulut tertutup)"),
    (78,  "MENGUAP #2 — buka mulut lebar ~1 dtk"),
    (83,  "NORMAL sebentar"),
    (86,  "MENGUAP #3 — buka mulut lebar ~1 dtk  (memicu YAWN ALERT ~88 dtk)"),
    (91,  "NORMAL"),
    (95,  "TUTUP MATA TERUS ~35 dtk  (MICROSLEEP ~110 dtk, lalu CRITICAL ~125 dtk)"),
    (130, "NORMAL — buka mata, kedip biasa (cek tidak ada alarm palsu)"),
    (150, "SELESAI — hentikan rekaman"),
]

# Ground-truth (harus konsisten dengan SCHEDULE). event_type,t_start_s,t_end_s
EVENTS = [
    ("microsleep", 45, 50),
    ("yawn_alert", 88, 95),
    ("microsleep", 110, 120),
    ("critical", 125, 130),
]

BELL = "\a"
BAR = "=" * 64

# Panduan "mau ngambil kondisi apa" — dua jenis sesi (lihat PROSEDUR_PENGAMBILAN_DATA_PI.md).
CONDITIONS = [
    ("— A. LATENSI / FPS (tanpa partisipan, ulangi 5×, TANPA --scripted) —", "", ""),
    ("lat_pi_picam", "Latensi Pi kamera CSI", "--condition lat_pi_picam --duration 30"),
    ("lat_pi_usb",   "Latensi Pi webcam USB", "--condition lat_pi_usb --camera usb --duration 30"),
    ("lat_net",      "Latensi network-send",  "--condition lat_net --bench_network 30 --duration 1"),
    ("— B. AKURASI (dengan partisipan, drowsiness DIPERANKAN, PAKAI --scripted) —", "", ""),
    ("p<NN>_<kam>_<jarak>_<lux>", "1 sesi terscript 2,5 mnt -> isi accuracy_%",
     "--scripted --condition p01_picam_60_150 --session_log sessions/p01_picam_60_150.csv"),
    ("  (kamera USB)", "sesi sama dgn webcam",
     "--scripted --camera usb --condition p01_usb_60_150 --session_log sessions/p01_usb_60_150.csv"),
]


def print_conditions():
    """Panduan singkat: kondisi mana yang harus diambil + contoh perintah."""
    print(f"\n{BAR}\nPANDUAN KONDISI (nilai --condition) — pilih sesuai tujuan\n{BAR}")
    print("Konvensi nama sesi akurasi: p<NN>_<kamera>_<jarak_cm>_<lux>  (mis. p03_usb_40_500)\n")
    for name, desc, ex in CONDITIONS:
        if not desc:                      # baris judul kelompok
            print(f"\n{name}")
            continue
        print(f"  {name:26s} {desc}")
        print(f"      python3 realtime_eval/measure_realtime_pi.py {ex}")
    print(f"\n{BAR}")
    print("Ingat: --scripted -> tampil aba-aba live + AUTO-ISI accuracy_% & false_alert_rate_%.")
    print("       tanpa --scripted -> hanya latensi/FPS (accuracy_% dibiarkan kosong).")
    print(f"{BAR}\n")


def write_truth(path: str):
    """Tulis ground-truth EVENTS ke CSV (untuk score_annotated_session.py manual)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "t_start_s", "t_end_s"])
        w.writerows(EVENTS)
    print(f"Ground-truth ditulis: {path} ({len(EVENTS)} event)")


def print_card():
    """Cetak kartu jadwal (aba-aba) + ground-truth — untuk dihafal/dicetak operator."""
    print(f"\n{BAR}\nJADWAL SESI TERSCRIPT (2,5 menit)\n{BAR}")
    for i, (t, instr) in enumerate(SCHEDULE):
        t_next = SCHEDULE[i + 1][0] if i + 1 < len(SCHEDULE) else t
        rng = f"{t:>3d}-{t_next:<3d}s" if i + 1 < len(SCHEDULE) else f"{t:>3d}s"
        print(f"  {rng:>10s} | {instr}")
    print(f"{BAR}\nGround-truth alert: " +
          ", ".join(f"{e}@{a}-{b}s" for e, a, b in EVENTS) + f"\n{BAR}\n")


class CuePlayer:
    """Tampilkan aba-aba SCHEDULE berdasar 'elapsed' dari loop perekaman (master clock),
    sehingga aba-aba SELALU sinkron dengan frame yang direkam."""
    def __init__(self, schedule, enabled=True):
        self.sch = schedule
        self.enabled = enabled
        self.idx = -1
        self._last_status_sec = -1
        self.total = schedule[-1][0]

    def update(self, elapsed: float):
        if not self.enabled:
            return
        new = self.idx
        while new + 1 < len(self.sch) and self.sch[new + 1][0] <= elapsed:
            new += 1
        if new != self.idx:
            self.idx = new
            print(f"\n{BELL}{BAR}\n  [{int(elapsed):3d}s]  {self.sch[self.idx][1]}\n{BAR}",
                  flush=True)
            self._last_status_sec = -1
        if int(elapsed) != self._last_status_sec and self.idx >= 0:
            self._last_status_sec = int(elapsed)
            t_next = self.sch[self.idx + 1][0] if self.idx + 1 < len(self.sch) else self.total
            remain = max(0.0, t_next - elapsed)
            sys.stdout.write(f"\r    t={int(elapsed):3d}s | aba-aba berikutnya dlm {remain:4.0f}s   ")
            sys.stdout.flush()


def autoscore(log_rows, duration_s):
    """Skor sesi terscript vs EVENTS -> {accuracy_pct, false_alert_pct, ...}.
    Memakai kembali engine temporal dari score_annotated_session.py (tanpa rekam ulang)."""
    if not log_rows:
        return None
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    try:
        import pandas as pd
        from score_annotated_session import DEFAULTS, score, simulate_engine
    except Exception as e:  # pandas / modul tak ada -> jangan gagalkan sesi
        print(f"[skor] auto-scoring dilewati ({e})")
        return None
    frames = pd.DataFrame(log_rows)
    truth = pd.DataFrame(EVENTS, columns=["event_type", "t_start_s", "t_end_s"])
    alerts = simulate_engine(frames, **DEFAULTS)
    s = score(alerts, truth, duration_s)
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    # accuracy_% = akurasi tingkat-event (CSI): dihukum oleh miss (FN) & alarm palsu (FP)
    s["accuracy_pct"] = round(100 * tp / (tp + fp + fn), 1) if (tp + fp + fn) else 0.0
    # false_alert_rate_% = porsi alert yang ternyata palsu
    s["false_alert_pct"] = round(100 * fp / (tp + fp), 1) if (tp + fp) else 0.0
    return s


class RpicamCapture:
    """Pipe MJPEG dari rpicam-vid (kamera CSI Pi). Berbeda dari _RpicamCapture di
    app2_rpi.py: --mode TIDAK dipaksa (mode '1536:864:10:P' hanya valid untuk
    sensor IMX708/Camera Module 3 dan membuat rpicam-vid mati di kamera lain),
    stderr rpicam DISIMPAN ke file untuk diagnosa, dan ada jeda inisialisasi sensor."""
    def __init__(self, width=640, height=480, framerate=15, mode=None, stderr_log=None):
        subprocess.run(["pkill", "-f", "rpicam-vid"], capture_output=True)
        time.sleep(0.3)
        cmd = ["rpicam-vid", "-t", "0", "--codec", "mjpeg", "--nopreview",
               "--width", str(width), "--height", str(height),
               "--framerate", str(framerate), "-o", "-"]
        if mode:
            cmd += ["--mode", mode]
        self._errpath = stderr_log
        self._err = open(stderr_log, "wb") if stderr_log else subprocess.DEVNULL
        print(f"[camera] rpicam-vid: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err)
        self._buf = b""
        time.sleep(1.0)  # beri sensor waktu inisialisasi sebelum baca pertama

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
        if self._err is not subprocess.DEVNULL:
            self._err.close()


class UsbCapture:
    """Webcam USB via OpenCV (untuk kamera ke-2 di protokol, atau bila CSI bermasalah)."""
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f"[ERROR] webcam USB index {index} tidak terbuka.")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


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
    ap.add_argument("--duration", type=float, default=None,
                    help="durasi detik (default: 30; --scripted otomatis jadi panjang jadwal+5)")
    ap.add_argument("--scripted", action="store_true",
                    help="tampilkan aba-aba SCHEDULE live + AUTO-ISI accuracy_%/false_alert_rate_% "
                         "(sesi akurasi dengan partisipan)")
    ap.add_argument("--no_cue", action="store_true",
                    help="dengan --scripted: tetap auto-skor tapi JANGAN tampilkan aba-aba di layar")
    ap.add_argument("--no_wait", action="store_true",
                    help="dengan --scripted: jangan tunggu ENTER sebelum mulai")
    ap.add_argument("--list_conditions", action="store_true",
                    help="cetak panduan kondisi (mau ambil kondisi apa) lalu keluar")
    ap.add_argument("--card_only", action="store_true", help="cetak kartu jadwal terscript lalu keluar")
    ap.add_argument("--write_truth", metavar="PATH", default=None,
                    help="tulis ground-truth (EVENTS) ke CSV lalu keluar")
    ap.add_argument("--session_log", default=None, help="CSV per-frame untuk score_annotated_session.py")
    ap.add_argument("--bench_network", type=int, default=0, help="jumlah POST untuk mengukur latensi network-send")
    ap.add_argument("--out_csv", default=os.path.join(THIS_DIR, "realtime_baseline_results.csv"))
    ap.add_argument("--eye_model", default=os.path.join(PROJECT_ROOT, "model_eye_mobilenet.tflite"))
    ap.add_argument("--mouth_model", default=os.path.join(PROJECT_ROOT, "model_mouth_mobilenet.tflite"))
    ap.add_argument("--camera", choices=["picam", "usb"], default="picam",
                    help="picam = kamera CSI (rpicam-vid); usb = webcam USB (OpenCV)")
    ap.add_argument("--camera_index", type=int, default=0, help="index webcam USB (--camera usb)")
    ap.add_argument("--rpicam_mode", default="",
                    help="mode sensor rpicam mis. '1536:864:10:P' (KOSONGKAN untuk mode default; "
                         "mode spesifik hanya valid untuk sensor tertentu spt IMX708)")
    args = ap.parse_args()

    # --- perintah info/util yang langsung keluar ---
    if args.list_conditions:
        print_conditions(); return
    if args.card_only:
        print_card(); return
    if args.write_truth:
        write_truth(args.write_truth); return

    # --scripted: durasi default = panjang jadwal + 5 dtk buffer; wajib rekam log per-frame
    if args.duration is None:
        args.duration = (SCHEDULE[-1][0] + 5) if args.scripted else 30.0
    need_log = args.session_log is not None or args.scripted

    net = bench_network(load_endpoint(), args.bench_network) if args.bench_network else None

    if args.camera == "usb":
        cap = UsbCapture(index=args.camera_index)
    else:
        rpicam_log = os.path.join(THIS_DIR, "rpicam_stderr.log")
        cap = RpicamCapture(mode=(args.rpicam_mode or None), stderr_log=rpicam_log)
    interp_e, ie, oe = load_tflite(args.eye_model)
    interp_m, im, om = load_tflite(args.mouth_model)
    face_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    stage = defaultdict(list)
    frame_lat, n_faces = [], 0
    cpu, ram = [], []
    log_rows = []
    proc = psutil.Process() if psutil else None

    # Sesi terscript: cetak kartu jadwal, tunggu partisipan siap, lalu hitung mundur.
    cue = CuePlayer(SCHEDULE, enabled=args.scripted and not args.no_cue)
    if args.scripted:
        print_card()
        if not args.no_wait and sys.stdin and sys.stdin.isatty():
            input("Tekan ENTER saat partisipan siap (kamera sudah menyala) ...")
        for c in range(5, 0, -1):
            print(f"  MULAI dalam {c} ...", flush=True)
            time.sleep(1.0)
        print(f"\n{BAR}\n  ▶▶▶  MULAI (t=0) — ikuti aba-aba di layar  ◀◀◀\n{BAR}", flush=True)

    t_start = time.time(); n_frames = 0

    print(f"[measure] kondisi={args.condition}, durasi={args.duration}s ...", flush=True)
    while time.time() - t_start < args.duration:
        cue.update(time.time() - t_start)
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
        if need_log:
            log_rows.append(dict(t_rel_s=round(time.time() - t_start, 3),
                                 face_detected=face_found, eye_label=eye_label,
                                 eye_conf=round(eye_conf, 4), mouth_label=mouth_label,
                                 mouth_conf=round(mouth_conf, 4)))
    cap.release()

    if n_frames == 0:
        print("\n[DIAGNOSA] 0 frame tertangkap — kamera tidak menghasilkan gambar.")
        if args.camera == "picam":
            log = os.path.join(THIS_DIR, "rpicam_stderr.log")
            if os.path.exists(log):
                print(f"  Pesan error rpicam-vid ({log}):")
                with open(log, "rb") as f:
                    tail = f.read()[-800:].decode(errors="replace").strip()
                for line in tail.splitlines()[-12:]:
                    print("    " + line)
            print("  Coba: (a) tes manual  rpicam-vid -t 2000 --codec mjpeg --nopreview "
                  "--width 640 --height 480 -o /tmp/t.mjpeg")
            print("        (b) jika kamera BUKAN IMX708, jangan pakai --rpicam_mode")
            print("        (c) atau pakai webcam USB:  --camera usb --camera_index 0")
        else:
            print("  Webcam USB tidak memberi frame — cek index (--camera_index) & koneksi.")
        raise SystemExit(1)

    dur = time.time() - t_start
    fps = n_frames / dur if dur else 0.0

    # Sesi terscript: skor otomatis vs ground-truth EVENTS -> isi accuracy_%/false_alert_rate_%.
    sc = autoscore(log_rows, dur) if args.scripted else None

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
        accuracy_pct=(sc["accuracy_pct"] if sc else ""),
        false_alert_rate_pct=(sc["false_alert_pct"] if sc else ""),
    )
    # kolom disamakan namanya dgn measure_realtime_baseline.py agar aggregate cocok
    row["face_detect_rate_%"] = row.pop("face_detect_rate_pct")
    row["avg_cpu_%"] = row.pop("avg_cpu_pct")
    row["accuracy_%"] = row.pop("accuracy_pct")
    row["false_alert_rate_%"] = row.pop("false_alert_rate_pct")

    print("\n=== RINGKASAN (Pi edge) ===")
    for k, v in row.items():
        print(f"  {k:24s}: {v}")
    print(f"  {'(catatan)':24s}: face_detect_rate_% = {n_faces}/{n_frames} frame ada wajah "
          f"(rendah = wajah sering hilang: cahaya/pose/jarak).")

    if sc is not None:
        print(f"\n{BAR}\n  SKOR AKURASI (vs ground-truth terscript, ambang produksi)\n{BAR}")
        print(f"  event benar (TP)      : {sc['tp']}/{sc['n_truth']}")
        print(f"  alarm palsu (FP)      : {sc['fp']}   miss (FN): {sc['fn']}")
        print(f"  sensitivity (recall)  : {sc['sensitivity']:.3f}" if sc['sensitivity'] == sc['sensitivity'] else "  sensitivity           : n/a")
        print(f"  precision             : {sc['precision']:.3f}" if sc['precision'] == sc['precision'] else "  precision              : n/a")
        print(f"  false alert / menit   : {sc['false_alerts_per_min']:.2f}")
        print(f"  -> accuracy_%={row['accuracy_%']}  false_alert_rate_%={row['false_alert_rate_%']}")
        print(f"{BAR}")
    elif args.scripted:
        print("\n[skor] tidak ada log untuk diskor (0 frame berlabel).")

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
