"""
Agregasi statistik atas beberapa sesi real-time yang sudah direkam ke
realtime_baseline_results.csv (lih. measure_realtime_baseline.py), untuk menjawab
permintaan reviewer (journal/revisi_from_reviewer.md, Reviewer 1 #7/#9 dan
Reviewer 2 #7): pengujian real-time berulang per kondisi, dengan mean + interval
kepercayaan, bukan cuma satu sesi per kondisi seperti draft awal (Table. 3).

Cara pakai:
  1. Jalankan measure_realtime_baseline.py beberapa KALI untuk kondisi yang sama
     (mis. 5 sesi terpisah untuk "normal_light_frontal"), supaya beberapa baris
     dengan condition sama terkumpul di realtime_baseline_results.csv.
  2. Jalankan skrip ini untuk mendapatkan mean, std, dan 95% CI (t-distribution,
     karena jumlah sesi biasanya kecil) per kondisi+platform untuk FPS, latensi,
     dan face-detection rate:

     python3 realtime_eval/aggregate_sessions.py
     python3 realtime_eval/aggregate_sessions.py --csv path/lain.csv
     python3 realtime_eval/aggregate_sessions.py --min_sessions 3

CATATAN JUJUR: skrip ini HANYA mengagregasi angka yang sudah terkumpul di CSV.
Kalau baru ada satu sesi per kondisi (kondisi awal repo ini), output akan
menunjukkan n=1 dan CI tidak dapat dihitung (NaN) -- itu tanda bahwa sesi
berulang secara fisik (kamera, subjek, kondisi pencahayaan) memang belum
dikumpulkan, dan itu perlu dilakukan di hardware sungguhan (bukan sesuatu yang
bisa disimulasikan dari sini).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

METRIC_COLUMNS = [
    "fps_overall", "avg_frame_latency_ms", "p95_frame_latency_ms",
    "face_detect_rate_%", "avg_cpu_%", "avg_ram_MB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agregasi mean + CI atas sesi real-time berulang")
    parser.add_argument("--csv", default=os.path.join(THIS_DIR, "realtime_baseline_results.csv"))
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--min_sessions", type=int, default=1,
                         help="Tampilkan peringatan untuk condition+platform dengan sesi < nilai ini")
    parser.add_argument("--out_csv", default=os.path.join(THIS_DIR, "realtime_sessions_aggregated.csv"))
    return parser.parse_args()


def mean_ci(values: np.ndarray, ci: float) -> tuple[float, float, float, float]:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    if n > 1:
        se = std / np.sqrt(n)
        tcrit = stats.t.ppf((1 + ci) / 2, df=n - 1)
        lo, hi = mean - tcrit * se, mean + tcrit * se
    else:
        lo, hi = float("nan"), float("nan")
    return mean, std, lo, hi


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"[ERROR] File tidak ditemukan: {args.csv}\n"
              f"Jalankan measure_realtime_baseline.py dulu untuk mengumpulkan sesi.")
        return

    df = pd.read_csv(args.csv)
    if df.empty:
        print(f"[ERROR] {args.csv} kosong.")
        return

    rows = []
    for (condition, platform), group in df.groupby(["condition", "platform"]):
        n_sessions = len(group)
        if n_sessions < args.min_sessions:
            print(f"[PERINGATAN] {condition} / {platform}: hanya {n_sessions} sesi "
                  f"(< {args.min_sessions}) -> CI belum bisa dipercaya, kumpulkan sesi lagi.")

        row = {"condition": condition, "platform": platform, "n_sessions": n_sessions}
        for col in METRIC_COLUMNS:
            if col not in group.columns:
                continue
            values = group[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean, std, lo, hi = mean_ci(values, args.ci)
            row[f"{col}_mean"] = round(mean, 3)
            row[f"{col}_std"] = round(std, 3) if not np.isnan(std) else None
            row[f"{col}_ci_lo"] = round(lo, 3) if not np.isnan(lo) else None
            row[f"{col}_ci_hi"] = round(hi, 3) if not np.isnan(hi) else None
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["platform", "condition"]).reset_index(drop=True)
    print(f"\n{'=' * 60}\nAGREGASI SESI REAL-TIME (mean + {int(args.ci * 100)}% CI, n_sessions per baris)\n{'=' * 60}")
    print(out_df.to_string(index=False))

    out_df.to_csv(args.out_csv, index=False)
    print(f"\nHasil agregasi disimpan ke: {args.out_csv}")


if __name__ == "__main__":
    main()
