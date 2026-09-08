#!/usr/bin/env python3
# ============================================================
#  SCRIPT PENGUJIAN CITRA MENTAH BARU (Out-of-Sample Test)
#  ----------------------------------------------------------
#  Membaca sebuah citra (.jpg/.png), mengekstrak 50 fitur DFT
#  (azimuthal average), lalu memprediksi status keasliannya
#  menggunakan model LightGBM yang sudah dilatih.
#
#  Model:  models/lightgbm_model.txt (hasil train_benchmark.py)
#  Fitur : scripts/feature_extraction.py (sama dgn saat training)
# ============================================================

import argparse
import os

import numpy as np
import lightgbm as lgb

# Import fungsi ekstraksi fitur yang konsisten dgn proses training.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features_from_image


# ============================================================
#  PENERJEMAHAN LABEL HASIL PREDIKSI
# ============================================================
def interpret(pred_class, prob_fake, prob_real):
    """
    Menerjemahkan hasil prediksi model menjadi kesimpulan forensik.
    Label: 0 = fake (termasuk copy-move, splicing, deepfake)
           1 = real (asli / tanpa manipulasi)
    """
    if pred_class == 0:
        status = "FAKE (Manipulated)"
    else:
        status = "REAL (Authentic)"
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi keaslian satu citra baru menggunakan LightGBM."
    )
    parser.add_argument(
        "image_path",
        help="Path ke citra yang ingin diuji (.jpg / .png / dll).",
    )
    parser.add_argument(
        "--model", default=os.path.join("models", "lightgbm_model.txt"),
        help="Path ke model LightGBM tersimpan (default: models/lightgbm_model.txt).",
    )
    parser.add_argument(
        "--num-features", type=int, default=50,
        help="Jumlah fitur DFT (harus sama dengan saat training, default: 50).",
    )
    args = parser.parse_args()

    # --- 1. Validasi keberadaan file gambar & model ---
    if not os.path.exists(args.image_path):
        print(f"ERROR: Citra tidak ditemukan: {args.image_path}")
        return
    if not os.path.exists(args.model):
        print(f"ERROR: Model tidak ditemukan: {args.model}")
        print("Jalankan dulu scripts/train_benchmark.py agar model tersimpan.")
        return

    # --- 2. Muat model LightGBM tersimpan ---
    model = lgb.Booster(model_file=args.model)
    print(f"[OK] Model dimuat: {args.model}")

    # --- 3. Ekstraksi 50 fitur DFT dari citra ---
    features = extract_features_from_image(args.image_path, N=args.num_features)
    # Model LightGBM mengharapkan input 2D (baris x fitur).
    X_sample = features.reshape(1, -1)
    print(f"[OK] Fitur DFT diekstrak: {X_sample.shape[1]} fitur dari {args.image_path}")

    # --- 4. Prediksi kelas & probabilitas ---
    #    Perilaku LightGBM Booster.predict untuk klasifikasi BINER:
    #      - menghasilkan vektor 1D berisi probabilitas kelas positif (real=1)
    #        untuk tiap sampel (nilai 0..1).
    #      - Bila multi-kelas, menghasilkan matriks [sampel, kelas].
    #    Kode di bawah menangani kedua kasus agar tetap fleksibel.
    raw = model.predict(X_sample)
    raw = np.asarray(raw)

    if raw.ndim == 2 and raw.shape[1] == 2:   # matriks prob per kelas
        proba = raw[0]                        # [P(fake=0), P(real=1)]
    elif raw.ndim == 1:                       # biner: 1D = P(real=1) per sampel
        prob_real = float(raw[0])
        proba = [1.0 - prob_real, prob_real]  # [P(fake=0), P(real=1)]
    else:
        raise ValueError(
            f"Bentuk output prediksi tidak dikenal: shape={raw.shape}. "
            "Periksa versi LightGBM / format model."
        )

    pred_class = int(np.argmax(proba))    # kelas dengan probabilitas tertinggi

    status = interpret(pred_class, proba[0], proba[1])

    # --- 5. Cetak hasil forensik ---
    print("\n" + "=" * 60)
    print("                 HASIL ANALISIS FORENSIK")
    print("=" * 60)
    print(f"  File citra        : {os.path.basename(args.image_path)}")
    print(f"  Jumlah fitur DFT  : {X_sample.shape[1]}")
    print(f"  Prob. FAKE (0)    : {proba[0]*100:.2f} %")
    print(f"  Prob. REAL (1)    : {proba[1]*100:.2f} %")
    print("-" * 60)
    print(f"  Kesimpulan        : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
