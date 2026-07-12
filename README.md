# PiDeploy — Deploy & Menjalankan di Raspberry Pi 5

Folder ini berisi **semua** yang perlu di-upload ke Raspberry Pi 5 untuk:
1. menjalankan aplikasi deteksi kantuk live (`app.py`), dan
2. mengumpulkan data evaluasi real-time yang diminta reviewer (latensi on-device,
   akurasi multi-partisipan, sensitivity ambang).

> Detail desain studi (jumlah partisipan, kamera, jarak, lux, skrip drowsiness)
> ada di `journal/RASPBERRY_PI_DATA_COLLECTION.md` di repo utama. File ini fokus
> ke **cara deploy & menjalankan**.

## Isi folder

```
PiDeploy/
├── app.py                     # aplikasi alert live (deployment sebenarnya)
├── model_eye_mobilenet.tflite      # model mata (augmented, sama dgn yg dipakai paper)
├── model_mouth_mobilenet.tflite    # model mulut (augmented)
├── .env.example                    # template konfigurasi -> salin jadi .env
├── requirements-pi.txt             # dependensi pip
├── README_DEPLOY.md                # file ini
├── realtime_eval/
│   ├── measure_realtime_pi.py      # ← DIJALANKAN untuk studi (latensi + log per-frame)
│   ├── score_annotated_session.py  # akurasi/false-alarm + sweep ambang
│   └── aggregate_sessions.py       # mean ± 95% CI antar sesi
└── sessions/
    └── truth_scripted.csv          # ground-truth (edit sesuai skrip Anda)
```

⚠️ **Jaga struktur folder ini.** `measure_realtime_pi.py` mencari model `.tflite`
di folder induk `realtime_eval/` (yaitu di `PiDeploy/`). Kalau file dipindah,
lewatkan path lewat `--eye_model` / `--mouth_model`.

---

## Langkah 1 — Transfer folder ke Pi

Dari komputer ini (macOS), kirim seluruh folder `PiDeploy/` ke Pi via `scp`
(ganti `pi@raspberrypi.local` dengan user/host Pi Anda):

```bash
# jalankan di terminal macOS, dari /Users/rafie/web/drowsiness
scp -r PiDeploy pi@raspberrypi.local:~/PiDeploy
```

Alternatif: rsync (lebih cepat untuk update berulang):
```bash
rsync -av --progress PiDeploy/ pi@raspberrypi.local:~/PiDeploy/
```

Atau salin via flashdisk / `git clone` bila repo sudah di GitHub.

---

## Langkah 2 — Instalasi di Pi

SSH ke Pi, lalu:

```bash
ssh pi@raspberrypi.local
cd ~/PiDeploy

# paket kamera sistem (bukan pip):
sudo apt update && sudo apt install -y python3-pip python3-venv rpicam-apps

# virtualenv + dependensi python:
python3 -m venv ~/pivenv
source ~/pivenv/bin/activate
pip install -r requirements-pi.txt
```

Jika `tflite-runtime` gagal terpasang, buka `requirements-pi.txt`, comment baris
`tflite-runtime`, uncomment `ai-edge-litert` (atau `tensorflow`), lalu ulangi
`pip install -r requirements-pi.txt`. Skrip mencoba ketiga runtime otomatis.

Cek kamera terdeteksi:
```bash
rpicam-hello -t 2000        # harus muncul preview singkat / tidak error
```

---

## Langkah 3 — Konfigurasi (.env)

Hanya perlu jika akan mengirim event ke backend (app live) atau mengukur latensi
network. Untuk pengukuran latensi/akurasi murni, LEWATI langkah ini.

```bash
cd ~/PiDeploy
cp .env.example .env
nano .env        # isi DROWSINESS_API_URL, DROWSINESS_DEVICE_TOKEN
```

---

## Langkah 4 — Smoke test (tanpa partisipan)

Ini sekaligus tes pertama `measure_realtime_pi.py` di hardware Pi:

```bash
cd ~/PiDeploy
source ~/pivenv/bin/activate
python3 realtime_eval/measure_realtime_pi.py --condition smoke --duration 15
```

Yang diharapkan: tercetak **RINGKASAN (Pi edge)** berisi FPS, breakdown latensi
per tahap (capture / face_detect / eye_infer / mouth_infer), face-detect rate,
CPU/RAM, dan satu baris `platform=edge` ditambahkan ke
`realtime_eval/realtime_baseline_results.csv`.

Kalau ada error import (mediapipe / tflite / rpicam), catat pesannya — biasanya
tinggal instal ulang runtime yang benar (lihat Troubleshooting).

Benchmark latensi network (opsional, butuh backend hidup + .env):
```bash
python3 realtime_eval/measure_realtime_pi.py --bench_network 30 --condition net_bench --duration 1
```

---

## Langkah 5 — Sesi pengukuran dengan partisipan (data untuk Table 6 & 7)

1. Pasang kamera di tripod, ukur jarak (mis. 60 cm), atur lampu ke lux target.
2. Beri partisipan kartu skrip (lihat jadwal di
   `journal/RASPBERRY_PI_DATA_COLLECTION.md`) dan minta persetujuan (drowsiness
   **diperankan**).
3. Sesuaikan `sessions/truth_scripted.csv` bila jadwal berbeda.
4. Rekam satu sesi (contoh: partisipan 01, kamera Pi, 60 cm, 150 lux):

```bash
python3 realtime_eval/measure_realtime_pi.py \
    --condition p01_picam_60cm_150lx \
    --duration 155 \
    --session_log sessions/p01_picam_60_150.csv
```

`--session_log` menyimpan label per-frame (mata/mulut + confidence) → dipakai
untuk skoring akurasi & sweep ambang tanpa merekam ulang.

---

## Langkah 6 — Ubah rekaman jadi angka (Table 6, Table 7, sensitivity)

```bash
# (a) Akurasi / sensitivity / false-alarm satu sesi (baris Table 6):
python3 realtime_eval/score_annotated_session.py \
    --frames sessions/p01_picam_60_150.csv \
    --truth  sessions/truth_scripted.csv

# (b) Sensitivity ambang — trade-off false-alert vs missed (REQ-1.6):
python3 realtime_eval/score_annotated_session.py \
    --frames sessions/p01_picam_60_150.csv \
    --truth  sessions/truth_scripted.csv --sweep

# (c) Setelah banyak sesi terkumpul: mean ± 95% CI per kondisi (Table 6/7 CI):
python3 realtime_eval/aggregate_sessions.py --min_sessions 3
```

Latensi per-tahap on-device (Table 7) sudah ada di setiap baris `platform=edge`
pada `realtime_eval/realtime_baseline_results.csv` (kolom `avg_face_detect_ms`,
`avg_eye_infer_ms`, `avg_mouth_infer_ms`, `network_send_avg_ms`, dll).

Verifikasi cepat logika skoring tanpa hardware (opsional, bisa di laptop):
```bash
python3 realtime_eval/score_annotated_session.py --selftest
```

---

## Langkah 7 — (Opsional) Jalankan aplikasi live

```bash
cd ~/PiDeploy
source ~/pivenv/bin/activate
python3 app.py
```

Berjalan headless otomatis bila tidak ada DISPLAY; event dikirim ke
`DROWSINESS_API_URL`, dan di-antre ke file pending saat offline lalu dikirim ulang.

---

## Langkah 8 — Ambil hasil kembali ke komputer

```bash
# di macOS:
scp -r pi@raspberrypi.local:~/PiDeploy/realtime_eval/realtime_baseline_results.csv ./
scp -r pi@raspberrypi.local:~/PiDeploy/sessions ./PiDeploy_sessions_backup
```

Lalu isi angka nyata ke **Table 6 & Table 7 (yang masih merah)** di
`journal/template2025-IJEEEMI_Rafie_REVISED_v3.docx`, hapus teks merahnya, dan
pindahkan REQ-2.1/2.3/2.4/1.6 ke ✅.

---

## Troubleshooting

| Gejala | Solusi |
|---|---|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` (di venv aktif) |
| `Didn't find op ... FULLY_CONNECTED v12` saat load .tflite | runtime TFLite terlalu lama; pakai `ai-edge-litert` atau `tensorflow` terbaru |
| `rpicam-vid: command not found` | `sudo apt install -y rpicam-apps` |
| Kamera tidak terbaca / frame kosong | cek kabel CSI; `rpicam-hello -t 2000`; pastikan tak ada proses `rpicam-vid` lain (skrip sudah `pkill` otomatis) |
| Model tidak ketemu | jalankan dari dalam `~/PiDeploy`, atau beri `--eye_model ~/PiDeploy/model_eye_mobilenet.tflite --mouth_model ...` |
| FPS jauh di bawah target 15 | pastikan active cooling terpasang; tutup app lain; cek suhu `vcgencmd measure_temp` |

## Catatan penting

- Model `.tflite` di folder ini adalah versi **augmented** (yang dilaporkan di
  paper). Bila Anda melatih ulang model di repo utama, **salin ulang** `.tflite`
  yang baru ke sini sebelum deploy.
- `measure_realtime_pi.py` belum pernah dijalankan di Pi asli dari sisi
  pengembangan (tidak ada hardware Pi di lingkungan build) — perlakukan smoke test
  Langkah 4 sebagai eksekusi pertamanya. Kode-nya meniru pipeline `app.py`
  yang sudah terbukti (rpicam → MediaPipe → TFLite).
