"""
Probe kamera + deteksi wajah — mendiagnosa face_detect_rate_% = 0.

Meniru PERSIS langkah deteksi wajah di measure_realtime_pi.py (flip -> BGR2RGB ->
MediaPipe FaceDetection) pada beberapa frame, lalu melaporkan:
  * berapa frame ter-decode (kamera memberi gambar?),
  * rata-rata brightness (mendekati 0 = frame HITAM: shutter/exposure/kamera salah),
  * berapa frame ADA wajah (0 = deteksi gagal, bukan skrip yang salah),
  * menyimpan 1 frame beranotasi ke --out supaya bisa DILIHAT langsung.

Jalankan (di Pi/desktop yg sama dgn recorder):
  # webcam USB index 0 (default measure_realtime_pi --camera usb):
  python3 realtime_eval/probe_camera.py --camera usb --camera_index 0
  # coba index lain bila 0 salah kamera:
  python3 realtime_eval/probe_camera.py --camera usb --camera_index 1
  # kamera CSI Pi:
  python3 realtime_eval/probe_camera.py --camera picam
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# Pakai ulang capture + mediapipe DARI recorder agar identik dgn kondisi nyata.
from measure_realtime_pi import RpicamCapture, UsbCapture, mp, cv2, THIS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", choices=["picam", "usb"], default="usb")
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--n", type=int, default=60, help="jumlah frame yang diperiksa")
    ap.add_argument("--rpicam_mode", default="")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "probe_frame.jpg"))
    args = ap.parse_args()

    if args.camera == "usb":
        cap = UsbCapture(index=args.camera_index)
    else:
        cap = RpicamCapture(mode=(args.rpicam_mode or None),
                            stderr_log=os.path.join(THIS_DIR, "rpicam_stderr.log"))
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)

    n_read = n_face = 0
    bright = []
    saved = False
    best = None  # (n_deteksi, frame_annotated) untuk disimpan
    print(f"[probe] memeriksa {args.n} frame dari kamera={args.camera} ...", flush=True)
    for _ in range(args.n):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        n_read += 1
        frame = cv2.flip(frame, 1)
        bright.append(float(frame.mean()))
        res = face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = res.detections or []
        if dets:
            n_face += 1
            fh, fw, _ = frame.shape
            for d in dets:
                b = d.location_data.relative_bounding_box
                x, y = int(b.xmin * fw), int(b.ymin * fh)
                cv2.rectangle(frame, (x, y),
                              (x + int(b.width * fw), y + int(b.height * fh)),
                              (0, 255, 0), 2)
        # simpan frame terbaik (paling banyak wajah); kalau tak ada, simpan yg terakhir
        score = len(dets)
        if best is None or score > best[0]:
            best = (score, frame.copy())
    cap.release()

    if best is not None:
        cv2.imwrite(args.out, best[1])
        saved = True

    print("\n=== HASIL PROBE ===")
    print(f"  frame ter-decode      : {n_read}/{args.n}")
    print(f"  brightness rata-rata  : {np.mean(bright):.1f} (0=hitam, ~100-160 normal)"
          if bright else "  brightness rata-rata  : n/a (tidak ada frame)")
    print(f"  frame ADA wajah       : {n_face}/{n_read} "
          f"({100*n_face/n_read:.0f}%)" if n_read else "  frame ADA wajah       : 0")
    if saved:
        print(f"  frame beranotasi      : {args.out}  <- BUKA & LIHAT")

    print("\n--- DIAGNOSA ---")
    if n_read == 0:
        print("  Kamera TIDAK memberi frame. USB: coba --camera_index 1/2, cek koneksi.")
    elif bright and np.mean(bright) < 15:
        print("  Frame nyaris HITAM. Salah kamera (index), shutter/tutup lensa, atau exposure.")
        print("  -> coba --camera_index lain, atau kamera CSI: --camera picam")
    elif n_face == 0:
        print("  Frame terang TAPI 0 wajah. Kemungkinan: wajah di luar frame / terlalu")
        print("  jauh/dekat/miring, ATAU kamera menghadap arah lain (index salah).")
        print("  -> buka frame beranotasi di atas; pastikan WAJAH terlihat & tegak.")
        print("  -> coba --camera_index lain; untuk USB pastikan bukan kamera IR/virtual.")
    else:
        print(f"  OK: deteksi wajah berjalan ({100*n_face/n_read:.0f}%). Jika saat sesi 0%,")
        print("  berarti index/kondisi kamera saat sesi berbeda dari probe ini.")


if __name__ == "__main__":
    main()
