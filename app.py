"""
Drowsiness Detection — Raspberry Pi (TFLite & MediaPipe Optimized)
===============================================================
Kode ini dioptimalkan untuk Raspberry Pi 5 menggunakan MediaPipe
untuk deteksi wajah yang toleran sudut, dan TFLite untuk deteksi mata/mulut.

PERBAIKAN: Menambahkan boundary box (kotak deteksi) pada gambar
yang dikirim ke server backend untuk validasi visual.

Dibutuhkan di Virtual Environment Pi (Python 3.10+):
    pip install mediapipe opencv-python tflite-runtime numpy

Jalankan:
    python app.py
"""

import base64
import cv2
import json
import numpy as np
import os
import subprocess
import time
from datetime import datetime
from urllib import error, request

# --- LOAD LIBRARY MEDIA PIPE ---
try:
    import mediapipe as mp
except ModuleNotFoundError:
    print("\n[ERROR] MediaPipe belum terinstal. Silakan jalankan: pip install mediapipe")
    exit(1)

# --- LOAD LIBRARY TFLITE ---
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    try:
        import ai_edge_litert.interpreter as tflite
    except ModuleNotFoundError:
        import tensorflow.lite as tflite


def load_env_file(file_name='.env'):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if line == '' or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key != '':
                os.environ.setdefault(key, value)

load_env_file()

# =========================
# CONFIG GLOBAL & BACKEND
# =========================
EYE_CLOSED_SECS_1 = 15.0       # durasi mata tertutup (detik) -> tier micro-sleep, sama seperti desktop Eq.(5)
EYE_CLOSED_SECS_2 = 30.0       # durasi mata tertutup (detik) -> tier kritis/tertidur, sama seperti desktop Eq.(5)
EYE_OPEN_RESET_SECS = 0.4      # toleransi kedip: mata harus terbuka terus selama ini sebelum timer direset
YAWN_LIMIT = 3                 # jumlah menguap dalam TIME_WINDOW
TIME_WINDOW = 60               # detik jendela penghitungan menguap
YAWN_FRAME_THRESHOLD = 5       # frame mulut terbuka berturut-turut untuk 1 yawn
MIN_CONFIDENCE_TFLITE = 0.75   # confidence minimum untuk model mata/mulut
MIN_FACE_DETECTION_CONF = 0.6  # confidence minimum MediaPipe menemukan wajah

BACKEND_EVENT_URL = os.getenv('DROWSINESS_API_URL', 'http://127.0.0.1:8000/api/realtime/kelelahan/events').strip()
BACKEND_DEVICE_TOKEN = os.getenv('DROWSINESS_DEVICE_TOKEN', '').strip()
SOURCE_DEVICE = os.getenv('DROWSINESS_SOURCE_DEVICE', 'raspberry-pi-5').strip()
PENDING_EVENTS_FILE = os.getenv('DROWSINESS_PENDING_FILE', 'pending_drowsiness_events.jsonl').strip()
EVENT_COOLDOWN_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 5

# Mode headless otomatis jika DISPLAY kosong (running di background)
HEADLESS = not bool(os.getenv('DISPLAY', '').strip())

# =========================
# STATE VARIABLES
# =========================
eye_closed_start = None        # timestamp saat mata mulai terklasifikasi Closed (None jika sedang Open)
eye_open_start = None          # timestamp saat mata mulai terklasifikasi Open lagi (untuk toleransi kedip)
eye_closed_duration = 0.0      # durasi mata tertutup berjalan (detik), dipakai untuk alert & payload
YAWN_COUNTER = 0
YAWN_COOLDOWN = False
yawn_frame_count = 0
last_yawn_time = time.time()
last_sent_at = {'micro_sleep': 0.0, 'yawn_alert': 0.0}
active_alerts = set()
_frame_count = 0
_last_heartbeat = 0.0
HEARTBEAT_INTERVAL = 10  # detik

# =========================
# 1. SETUP KAMERA RPI 5
# =========================
class _RpicamCapture:
    """Baca frame dari rpicam-vid via MJPEG pipe."""
    def __init__(self, width=640, height=480):
        # Bersihkan proses lama
        subprocess.run(['pkill', '-f', 'rpicam-vid'], capture_output=True)
        time.sleep(0.2)
        cmd = [
            'rpicam-vid', '-t', '0', '--codec', 'mjpeg', '--nopreview',
            '--width', str(width), '--height', str(height), '--framerate', '15',
            '--mode', '1536:864:10:P',  # Mode optimal Pi 5
            '-o', '-',
        ]
        # DEVNULL agar tidak memenuhi terminal dengan log rpicam-vid
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._buf = b''
        print(f'rpicam-vid started (PID: {self._proc.pid})', flush=True)

    def read(self):
        while True:
            if self._proc.poll() is not None:
                return False, None
            chunk = self._proc.stdout.read(8192)
            if not chunk:
                return False, None
            self._buf += chunk
            start = self._buf.find(b'\xff\xd8')
            end = self._buf.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame

    def release(self):
        self._proc.terminate()
        self._proc.wait()

print("Kamera: rpicam-vid pipe (Arducam/IMX708 optimized)")
cap = _RpicamCapture(640, 480)

# =========================
# 2. SETUP MODEL TFLITE
# =========================
def load_tflite(model_path):
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

try:
    interp_eye, in_eye, out_eye = load_tflite('model_eye_mobilenet.tflite')
    interp_mouth, in_mouth, out_mouth = load_tflite('model_mouth_mobilenet.tflite')
    eye_labels = ['Closed', 'Open']
    mouth_labels = ['No_yawn', 'Yawn']
    print("Model TFLite mata & mulut berhasil dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal memuat model TFLite: {e}")
    cap.release()
    exit(1)

# =========================
# 3. SETUP MODEL MEDIAPIPE (DETEKSI WAJAH)
# =========================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, # Wajah dekat (< 2 meter)
    min_detection_confidence=MIN_FACE_DETECTION_CONF
)
print(f"MediaPipe Face Detection aktif (Conf Threshold: {MIN_FACE_DETECTION_CONF})")

# =========================
# UTILITY FUNCTIONS
# =========================

def get_prediction_tflite(img, interp, input_details, output_details, labels, threshold, safe_index):
    """Menjalankan inferensi TFLite untuk ROI mata/mulut."""
    try:
        if img.size == 0: return labels[safe_index], 0.0
        # Prapemrosesan: Resize ke 160x160, normalisasi 0-1
        img_resized = cv2.resize(img, (160, 160)).astype("float32") / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        interp.set_tensor(input_details[0]['index'], img_resized)
        interp.invoke()
        pred = interp.get_tensor(output_details[0]['index'])[0]

        idx = np.argmax(pred)
        conf = pred[idx]

        if conf < threshold:
            return labels[safe_index], conf
        return labels[idx], conf
    except Exception:
        return labels[safe_index], 0.0

def draw_annotations_for_server(frame, face_coords, eye_roi_coords, mouth_roi_coords, label_eye, label_mouth, status_text, alert_color):
    """
    PERBAIKAN: Menggambar boundary box dan status khusus pada frame
    yang akan dikirim ke server, meskipun menjalankan mode HEADLESS.
    """
    annotated_frame = frame.copy()
    fh, fw, _ = annotated_frame.shape

    # 1. Gambar Kotak Wajah (Biru)
    fx, fy, fw_box, fh_box = face_coords
    cv2.rectangle(annotated_frame, (fx, fy), (fx + fw_box, fy + fh_box), (255, 0, 0), 2)
    cv2.putText(annotated_frame, "WAJAH", (fx, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 2. Gambar Kotak Mata (Merah jika Tertutup, Hijau jika Terbuka)
    ex, ey, ew, eh = eye_roi_coords
    e_color = (0, 0, 255) if label_eye == 'Closed' else (0, 255, 0)
    cv2.rectangle(annotated_frame, (ex, ey), (ex + ew, ey + eh), e_color, 2)
    cv2.putText(annotated_frame, f"MATA: {label_eye}", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, e_color, 1)

    # 3. Gambar Kotak Mulut (Oranye/Merah jika Menguap)
    mx, my, mw, mh = mouth_roi_coords
    m_color = (0, 165, 255) if label_mouth == 'Yawn' else (255, 255, 0)
    cv2.rectangle(annotated_frame, (mx, my), (mx + mw, my + mh), m_color, 2)
    cv2.putText(annotated_frame, f"MULUT: {label_mouth}", (mx, my + mh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, m_color, 1)

    # 4. Gambar Dashboard Status di Atas Frame
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
    
    cv2.putText(annotated_frame, status_text, (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, alert_color, 2)
    cv2.putText(annotated_frame, f"Yawn: {YAWN_COUNTER}/{YAWN_LIMIT} | Closed: {eye_closed_duration:.1f}s | Waktu: {datetime.now().strftime('%H:%M:%S')}",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame

def encode_frame_jpeg(frame) -> str | None:
    """Resize frame (640x360), encode sebagai JPEG 60%, kembalikan string base64."""
    try:
        if frame is None: return None
        small = cv2.resize(frame, (640, 360))
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else None
    except Exception:
        return None

def store_pending_event_no_image(payload):
    """Simpan event tanpa image ke JSONL jika backend mati."""
    if not os.getenv('DROWSINESS_PENDING_FILE'): return
    try:
        payload_copy = {k: v for k, v in payload.items() if k != 'image_base64'}
        with open(PENDING_EVENTS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload_copy) + '\n')
    except Exception: pass

def send_event_payload(payload):
    """Mengirim payload event ke API Backend via HTTP POST."""
    if not BACKEND_EVENT_URL or not BACKEND_DEVICE_TOKEN: return False
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(
        BACKEND_EVENT_URL, data=data,
        headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'X-Device-Token': BACKEND_DEVICE_TOKEN},
        method='POST'
    )
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return 200 <= resp.status < 300
    except Exception: return False

def build_event_payload(event_type, severity, status_text, eye_label, conf_eye, mouth_label, conf_mouth, image_base64=None):
    payload = {
        'event_type': event_type, 'severity': severity, 'status_text': status_text,
        'eye_label': eye_label, 'mouth_label': mouth_label,
        'eye_confidence': round(float(conf_eye), 4), 'mouth_confidence': round(float(conf_mouth), 4),
        'eye_closed_duration_seconds': round(float(eye_closed_duration), 2), 'yawn_count': int(YAWN_COUNTER),
        'source_device': SOURCE_DEVICE, 'detected_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'metadata': {
            'time_window_seconds': TIME_WINDOW,
            'eye_closed_secs_1': EYE_CLOSED_SECS_1, 'eye_closed_secs_2': EYE_CLOSED_SECS_2,
            'yawn_limit': YAWN_LIMIT,
        }
    }
    if image_base64 is not None: payload['image_base64'] = image_base64
    return payload

def maybe_send_event(event_type, severity, status_text, eye_label, conf_eye, mouth_label, conf_mouth, current_time, image_to_send=None):
    """Memeriksa cooldown sebelum mengirim event."""
    if (current_time - last_sent_at[event_type]) < EVENT_COOLDOWN_SECONDS: return
    
    # Encode gambar yang sudah di-annotated
    image_b64 = encode_frame_jpeg(image_to_send)
    payload = build_event_payload(event_type, severity, status_text, eye_label, conf_eye, mouth_label, conf_mouth, image_b64)
    
    # Kirim asinkron (simulasi sederhana dengan pengecekan return)
    if not send_event_payload(payload):
        store_pending_event_no_image(payload)
        
    last_sent_at[event_type] = current_time

# =========================
# MAIN LOOP
# =========================
try:
    if HEADLESS:
        print(f"Sistem Berjalan (HEADLESS=True)... Tekan Ctrl+C untuk berhenti.")
    else:
        print("Sistem Berjalan (GUI)... Tekan 'q' untuk berhenti.")

    print("Memulai loop kamera...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            if HEADLESS: time.sleep(0.1); continue
            break

        frame = cv2.flip(frame, 1) # Mirror horizontal

        # --- DETEKSI WAJAH MEDIAPIPE (TOLERAN ANGLE) ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        num_faces = len(results.detections) if results.detections else 0

        current_time = time.time()
        _frame_count += 1

        # Reset hitungan menguap jika jendela waktu habis
        if current_time - last_yawn_time > TIME_WINDOW:
            YAWN_COUNTER = 0
            last_yawn_time = current_time

        status_text = "Status: Siaga (Aman)"
        alert_color = (0, 255, 0) # Hijau

        # Log Heartbeat berkala di terminal
        if current_time - _last_heartbeat >= HEARTBEAT_INTERVAL:
            _last_heartbeat = current_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] frame={_frame_count} | wajah={num_faces} | {status_text}", flush=True)

        if num_faces == 0:
            eye_closed_start = None
            eye_open_start = None
            eye_closed_duration = 0.0
            yawn_frame_count = 0
            YAWN_COOLDOWN = False

        pending_events_to_send = []

        # Proses wajah yang ditemukan
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                fh_frame, fw_frame, _ = frame.shape
                
                # Koordinat Bounding Box Wajah (di dalam frame)
                fx = max(0, int(bbox.xmin * fw_frame))
                fy = max(0, int(bbox.ymin * fh_frame))
                fw_box = min(fw_frame - fx, int(bbox.width * fw_frame))
                fh_box = min(fh_frame - fy, int(bbox.height * fh_frame))

                if fw_box < 50 or fh_box < 50: continue # Skip wajah terlalu kecil

                # --- Tentukan Koordinat ROI Mata & Mulut ---
                # ROI Mata (Area atas wajah, 50% tinggi)
                ex, ey = fx, fy
                ew, eh = fw_box, int(fh_box * 0.5)
                eye_roi = frame[ey:ey+eh, ex:ex+ew]

                # ROI Mulut (Area bawah wajah, antara 70%-95% tinggi)
                mx, my = fx + int(fw_box * 0.2), fy + int(fh_box * 0.7)
                mw, mh = int(fw_box * 0.6), int(fh_box * 0.25)
                mouth_roi = frame[my:my+mh, mx:mx+mw]

                # --- Prediksi TFLite (MobileNet) ---
                label_eye, conf_eye = get_prediction_tflite(
                    eye_roi, interp_eye, in_eye, out_eye, eye_labels, MIN_CONFIDENCE_TFLITE, safe_index=1
                )
                label_mouth, conf_mouth = get_prediction_tflite(
                    mouth_roi, interp_mouth, in_mouth, out_mouth, mouth_labels, MIN_CONFIDENCE_TFLITE, safe_index=0
                )

                # --- Logika Deteksi Drowsiness ---
                # Durasi wall-clock (bukan hitungan frame) agar tier alert konsisten
                # walau FPS Pi turun di bawah beban (lih. Eq. (5)/(11) & Bagian IV-A paper).
                if label_eye == 'Closed':
                    eye_open_start = None
                    if eye_closed_start is None:
                        eye_closed_start = current_time
                    eye_closed_duration = current_time - eye_closed_start
                else:
                    if eye_open_start is None:
                        eye_open_start = current_time
                    open_duration = current_time - eye_open_start
                    if open_duration >= EYE_OPEN_RESET_SECS:
                        eye_closed_start = None
                        eye_closed_duration = 0.0
                        eye_open_start = None
                    else:
                        eye_closed_duration = (
                            (current_time - eye_closed_start) if eye_closed_start else 0.0
                        )

                if label_mouth == 'Yawn':
                    yawn_frame_count += 1
                else:
                    yawn_frame_count = 0

                if yawn_frame_count >= YAWN_FRAME_THRESHOLD:
                    if not YAWN_COOLDOWN:
                        YAWN_COUNTER += 1
                        YAWN_COOLDOWN = True
                else:
                    YAWN_COOLDOWN = False

                # --- Tentukan Status Akhir (dua tier durasi, sama seperti desktop Eq.(5)) ---
                is_alert = False
                if eye_closed_duration >= EYE_CLOSED_SECS_2:
                    status_text = f"!!! BAHAYA: TERTIDUR ({eye_closed_duration:.1f}s) !!!"
                    alert_color = (0, 0, 180)  # Merah tua
                    is_alert = True
                elif eye_closed_duration >= EYE_CLOSED_SECS_1:
                    status_text = f"!!! BAHAYA: MICRO-SLEEP ({eye_closed_duration:.1f}s) !!!"
                    alert_color = (0, 0, 255) # Merah
                    is_alert = True
                elif YAWN_COUNTER >= YAWN_LIMIT:
                    status_text = f"PERINGATAN: LELAH ({YAWN_COUNTER}x Menguap)"
                    alert_color = (0, 165, 255) # Oranye
                    is_alert = True

                # --- JIKA TERJADI ALERT, BUAT GAMBAR ANNOTATED UNTUK SERVER ---
                annotated_image_for_server = None
                if is_alert:
                    # Simpan data koordinat untuk digambar
                    face_coords = (fx, fy, fw_box, fh_box)
                    eye_roi_coords = (ex, ey, ew, eh)
                    mouth_roi_coords = (mx, my, mw, mh)
                    
                    # Buat gambar khusus dengan boundary box
                    annotated_image_for_server = draw_annotations_for_server(
                        frame, face_coords, eye_roi_coords, mouth_roi_coords,
                        label_eye, label_mouth, status_text, alert_color
                    )

                # --- Kumpulkan Event yang Perlu Dikirim ---
                if eye_closed_duration >= EYE_CLOSED_SECS_2:
                    active_alerts.discard('micro_sleep')
                    if 'fatigue_alert' not in active_alerts:
                        pending_events_to_send.append(('fatigue_alert', 'danger', status_text, label_eye, conf_eye, label_mouth, conf_mouth, annotated_image_for_server))
                        active_alerts.add('fatigue_alert')
                elif eye_closed_duration >= EYE_CLOSED_SECS_1:
                    active_alerts.discard('fatigue_alert')
                    if 'micro_sleep' not in active_alerts:
                        pending_events_to_send.append(('micro_sleep', 'danger', status_text, label_eye, conf_eye, label_mouth, conf_mouth, annotated_image_for_server))
                        active_alerts.add('micro_sleep')
                else:
                    active_alerts.discard('micro_sleep')
                    active_alerts.discard('fatigue_alert')

                if YAWN_COUNTER >= YAWN_LIMIT:
                    if 'yawn_alert' not in active_alerts:
                        pending_events_to_send.append(('yawn_alert', 'warning', status_text, label_eye, conf_eye, label_mouth, conf_mouth, annotated_image_for_server))
                        active_alerts.add('yawn_alert')
                else: active_alerts.discard('yawn_alert')

        # --- Kirim Event ke Backend (Satu per Satu) ---
        for evt_data in pending_events_to_send:
            # Perhatikan: evt_data[-1] berisi annotated_image_for_server
            maybe_send_event(*evt_data[:-1], current_time, image_to_send=evt_data[-1])

        # Print alert di terminal jika HEADLESS
        if pending_events_to_send and HEADLESS:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] EVENT TERKIRIM: {status_text} | Wajah=Terdeteksi")

        # --- Tampilkan GUI (Hanya jika DISPLAY tersedia) ---
        if not HEADLESS:
            # Gambar visualisasi ringan di monitor lokal (opsional)
            cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, alert_color, 2)
            cv2.imshow('Drowsiness Detection RPi 5', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nDihentikan oleh pengguna.")

# cleanup
cap.release()
face_detection.close()
if not HEADLESS:
    cv2.destroyAllWindows() 