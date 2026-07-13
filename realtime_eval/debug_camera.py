"""
Diagnosa cepat kamera + face-detection untuk measure_realtime_pi.py.

Kalau measure_realtime_pi.py melaporkan face_detect_rate_% = 0 (dan eye/mouth
infer = 0), jalankan ini untuk tahu SEBABNYA: frame kosong? kamera salah?
MediaPipe tak menemukan wajah?

  python realtime_eval/debug_camera.py --camera usb --camera_index 0
  python realtime_eval/debug_camera.py --camera usb --camera_index 1
  python realtime_eval/debug_camera.py --camera picam

Output: shape frame, kecerahan rata-rata (0=hitam, 255=putih), berapa dari N
frame yang terdeteksi wajah, dan menyimpan 1 frame contoh ke
realtime_eval/debug_frame.jpg supaya bisa dilihat apa yang sebenarnya ditangkap.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import mediapipe as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def open_usb(index):
    # SAMA seperti app.py yang sudah terbukti jalan: pakai backend V4L2 eksplisit.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", choices=["usb", "picam"], default="usb")
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--n", type=int, default=60, help="jumlah frame untuk dicek")
    args = ap.parse_args()

    if args.camera == "usb":
        cap = open_usb(args.camera_index)
        if not cap.isOpened():
            raise SystemExit(f"[ERROR] webcam USB index {args.camera_index} tidak terbuka. "
                             "Coba index lain (0/1/2) atau cek `ls /dev/video*`.")
        read = cap.read
        release = cap.release
    else:
        import measure_realtime_pi as m  # pakai RpicamCapture yang sama
        c = m.RpicamCapture(stderr_log=os.path.join(THIS_DIR, "rpicam_stderr.log"))
        read, release = c.read, c.release

    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)

    n_read = n_face = 0
    sample_saved = False
    brightness = []
    for _ in range(args.n):
        ret, frame = read()
        if not ret or frame is None:
            print("[warn] read() gagal / frame None"); break
        n_read += 1
        brightness.append(float(frame.mean()))
        frame = cv2.flip(frame, 1)
        res = face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            n_face += 1
            if not sample_saved:  # simpan frame beranotasi pertama yg ada wajahnya
                d = res.detections[0].location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(d.xmin * w), int(d.ymin * h)
                cv2.rectangle(frame, (x, y), (x + int(d.width * w), y + int(d.height * h)),
                              (0, 255, 0), 2)
                cv2.imwrite(os.path.join(THIS_DIR, "debug_frame.jpg"), frame)
                sample_saved = True
    if not sample_saved and n_read:  # tak ada wajah -> simpan frame terakhir apa adanya
        cv2.imwrite(os.path.join(THIS_DIR, "debug_frame.jpg"), frame)
    release()

    print("\n=== DIAGNOSA KAMERA ===")
    print(f"  kamera            : {args.camera} (index {args.camera_index})")
    print(f"  frame terbaca     : {n_read}/{args.n}")
    print(f"  shape frame       : {frame.shape if n_read else 'N/A'}")
    print(f"  kecerahan rata2   : {np.mean(brightness):.1f} (0=hitam, 255=putih)"
          if brightness else "  kecerahan rata2   : N/A")
    print(f"  wajah terdeteksi  : {n_face}/{n_read} "
          f"({100*n_face/n_read:.0f}%)" if n_read else "  wajah terdeteksi  : N/A")
    print(f"  frame contoh      : {os.path.join(THIS_DIR, 'debug_frame.jpg')}")
    print("\nInterpretasi:")
    print("  - kecerahan ~0     -> kamera hitam/tertutup atau device salah")
    print("  - kecerahan wajar tapi wajah 0% -> kamera tak mengarah ke Anda / index salah")
    print("  - wajah ~100%      -> kamera OK; masalah measure ada di tempat lain")


if __name__ == "__main__":
    main()
