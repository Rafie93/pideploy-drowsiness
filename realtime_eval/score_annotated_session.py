"""
Skoring sesi real-time terhadap ground-truth beranotasi + analisis sensitivitas
ambang (menjawab REQ-2.1 accuracy/false-alarm dengan GT, dan REQ-1.6 sensitivity
analysis untuk ambang temporal — keduanya dari SATU rekaman per sesi).

Idenya: measure_realtime_pi.py (atau versi desktop) menuliskan LOG PER-FRAME
(timestamp, label mata/mulut + confidence) selama sesi. Skrip ini:
  1. Me-*re-simulasi* mesin keputusan temporal (durasi mata tertutup θ1=15s/θ2=30s
     dengan toleransi kedip, yawn = N frame beruntun, alert = K yawn / 60s) di atas
     log per-frame itu — sehingga ambang bisa diubah TANPA merekam ulang.
  2. Mencocokkan alert sistem dengan timeline ground-truth (skrip drowsiness yang
     diperankan partisipan) → TP/FP/FN → sensitivity (recall), precision,
     false-alarm rate (per menit), dan akurasi event.
  3. --sweep: menjalankan (1)-(2) untuk banyak kombinasi ambang dan mencetak
     trade-off false-alert vs missed-detection (langsung menjawab REQ-1.6).

Format file:
  --frames  CSV per-frame  : kolom  t_rel_s, eye_label, eye_conf, mouth_label, mouth_conf
                             (face_detected opsional; frame tanpa wajah -> perlakukan Open/No_yawn)
  --truth   CSV ground-truth: kolom  event_type, t_start_s, t_end_s
                             event_type ∈ {microsleep, critical, yawn_alert}
                             (jendela waktu saat alert jenis itu SEHARUSNYA muncul)

Contoh:
  python realtime_eval/score_annotated_session.py --frames sesi01_frames.csv --truth sesi01_truth.csv
  python realtime_eval/score_annotated_session.py --frames sesi01_frames.csv --truth sesi01_truth.csv --sweep
  python realtime_eval/score_annotated_session.py --selftest      # data sintetis, untuk verifikasi logika
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

# Default ambang = nilai produksi (lihat FatigueDesktop / app2_rpi.py Eq. (5)/(6)).
DEFAULTS = dict(theta1=15.0, theta2=30.0, blink_tol=0.4,
                yawn_frames=5, yawn_limit=3, yawn_window=60.0)
MATCH_TOL_S = 3.0  # alert dianggap cocok dgn GT bila jatuh di [t_start-… , t_end+tol]


def simulate_engine(frames: pd.DataFrame, theta1, theta2, blink_tol,
                    yawn_frames, yawn_limit, yawn_window) -> list[dict]:
    """Kembalikan daftar alert {type, t}: 'microsleep'(θ1), 'critical'(θ2), 'yawn_alert'."""
    alerts = []
    eye_closed_start = None      # kapan episode tutup mata mulai
    eye_open_since = None        # kapan mulai Open (untuk toleransi kedip)
    fired_micro = fired_crit = False
    consec_yawn = 0
    yawn_times: list[float] = []
    fired_yawn_window_until = -1.0

    for _, r in frames.iterrows():
        t = float(r["t_rel_s"])
        eye = str(r.get("eye_label", "Open"))
        mouth = str(r.get("mouth_label", "No_yawn"))

        # ---- logika mata: durasi tutup dgn toleransi kedip ----
        if eye == "Closed":
            eye_open_since = None
            if eye_closed_start is None:
                eye_closed_start = t
                fired_micro = fired_crit = False
        else:  # Open
            if eye_closed_start is not None:
                if eye_open_since is None:
                    eye_open_since = t
                # Open bertahan lebih lama dari toleransi kedip -> reset episode
                if t - eye_open_since >= blink_tol:
                    eye_closed_start = None
                    eye_open_since = None
                    fired_micro = fired_crit = False
        if eye_closed_start is not None:
            dur = t - eye_closed_start
            if dur >= theta1 and not fired_micro:
                alerts.append({"type": "microsleep", "t": t}); fired_micro = True
            if dur >= theta2 and not fired_crit:
                alerts.append({"type": "critical", "t": t}); fired_crit = True

        # ---- logika yawn: N frame beruntun = 1 yawn; K yawn / window = alert ----
        if mouth == "Yawn":
            consec_yawn += 1
            if consec_yawn == yawn_frames:
                yawn_times.append(t)
        else:
            consec_yawn = 0
        yawn_times = [yt for yt in yawn_times if t - yt <= yawn_window]
        if len(yawn_times) >= yawn_limit and t >= fired_yawn_window_until:
            alerts.append({"type": "yawn_alert", "t": t})
            fired_yawn_window_until = t + yawn_window  # jangan spam dalam window yg sama
            yawn_times = []
    return alerts


def score(alerts, truth: pd.DataFrame, duration_s: float, tol=MATCH_TOL_S) -> dict:
    gt = truth.to_dict("records")
    matched_gt = set()
    tp = fp = 0
    for a in alerts:
        hit = None
        for i, g in enumerate(gt):
            if g["event_type"] != a["type"]:
                continue
            if (g["t_start_s"] - tol) <= a["t"] <= (g["t_end_s"] + tol) and i not in matched_gt:
                hit = i; break
        if hit is not None:
            matched_gt.add(hit); tp += 1
        else:
            fp += 1
    fn = len(gt) - len(matched_gt)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    far_per_min = fp / (duration_s / 60.0) if duration_s else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, sensitivity=sens, precision=prec,
                false_alerts_per_min=far_per_min, n_alerts=len(alerts), n_truth=len(gt))


def run_once(frames, truth, params) -> dict:
    alerts = simulate_engine(frames, **params)
    dur = float(frames["t_rel_s"].max()) if len(frames) else 0.0
    return score(alerts, truth, dur)


def sweep(frames, truth):
    grid = dict(
        theta1=[10, 12, 15, 18, 20],
        theta2=[25, 30, 35],
        yawn_frames=[3, 5, 7],
        yawn_limit=[2, 3, 4],
    )
    rows = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        p = dict(DEFAULTS); p.update(dict(zip(keys, combo)))
        s = run_once(frames, truth, p)
        rows.append({**{k: p[k] for k in keys}, **s})
    return pd.DataFrame(rows)


def _selftest():
    """Bangun log per-frame sintetis + GT, lalu cek engine mendeteksi dengan benar."""
    fps, T = 12.0, 120.0
    ts = np.arange(0, T, 1 / fps)
    eye = np.array(["Open"] * len(ts), dtype=object)
    mouth = np.array(["No_yawn"] * len(ts), dtype=object)
    # microsleep: mata tertutup 30..50s (20s -> lewati θ1=15 dan θ2=30? 20<30, jadi hanya microsleep)
    eye[(ts >= 30) & (ts < 50)] = "Closed"
    # critical: tertutup 70..105s (35s -> lewati θ1 & θ2)
    eye[(ts >= 70) & (ts < 105)] = "Closed"
    # yawn: 3 yawn beruntun sekitar 10..25s (tiap yawn ~1s = 12 frame > 5)
    for c in (10, 15, 20):
        mouth[(ts >= c) & (ts < c + 1.0)] = "Yawn"
    frames = pd.DataFrame({"t_rel_s": ts, "eye_label": eye, "eye_conf": 0.9,
                           "mouth_label": mouth, "mouth_conf": 0.9})
    truth = pd.DataFrame([
        {"event_type": "yawn_alert", "t_start_s": 20, "t_end_s": 26},
        {"event_type": "microsleep", "t_start_s": 45, "t_end_s": 50},
        {"event_type": "microsleep", "t_start_s": 85, "t_end_s": 90},
        {"event_type": "critical", "t_start_s": 100, "t_end_s": 105},
    ])
    print("=== SELF-TEST (default thresholds) ===")
    print(run_once(frames, truth, dict(DEFAULTS)))
    print("\n=== SELF-TEST sweep (head) ===")
    sw = sweep(frames, truth)
    print(sw.sort_values(["fn", "fp"]).head(8).to_string(index=False))
    return frames, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames"); ap.add_argument("--truth")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out_csv", default=None)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    if not (args.frames and args.truth):
        ap.error("--frames dan --truth wajib (atau pakai --selftest)")
    frames = pd.read_csv(args.frames)
    truth = pd.read_csv(args.truth)
    params = {k: (int(getattr(args, k)) if k in ("yawn_frames", "yawn_limit") else getattr(args, k))
              for k in DEFAULTS}

    if args.sweep:
        df = sweep(frames, truth)
        out = args.out_csv or (os.path.splitext(args.frames)[0] + "_sweep.csv")
        df.to_csv(out, index=False)
        print(df.sort_values(["fn", "false_alerts_per_min"]).to_string(index=False))
        print(f"\nSweep disimpan ke: {out}")
    else:
        print(run_once(frames, truth, params))


if __name__ == "__main__":
    main()
