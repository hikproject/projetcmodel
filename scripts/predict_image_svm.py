#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import extract_features_from_image


def interpret(pred_class):
    return "FAKE (Manipulated)" if pred_class == 0 else "REAL (Authentic)"


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi satu citra baru menggunakan model SVM."
    )
    parser.add_argument(
        "image_path",
        help="Path ke citra yang ingin diuji (.jpg / .png / dll).",
    )
    parser.add_argument(
        "--model",
        default=os.path.join("models", "svm_model.pkl"),
        help="Path ke model SVM (.pkl).",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=50,
        help="Jumlah fitur DFT (harus sama dengan saat training, default: 50).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"ERROR: Citra tidak ditemukan: {args.image_path}")
        return

    if not os.path.exists(args.model):
        print(f"ERROR: Model tidak ditemukan: {args.model}")
        print("Jalankan dulu scripts/train_benchmark.py agar model tersimpan.")
        return

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print(f"[OK] Model SVM dimuat: {args.model}")

    features = extract_features_from_image(args.image_path, N=args.num_features)
    X_sample = features.reshape(1, -1)

    pred_class = int(model.predict(X_sample)[0])
    proba = model.predict_proba(X_sample)[0]

    # proba biasanya urutannya [P(fake=0), P(real=1)]
    fake_prob = float(proba[0])
    real_prob = float(proba[1])

    print("\n" + "=" * 60)
    print("                 HASIL ANALISIS FORENSIK (SVM)")
    print("=" * 60)
    print(f"  File citra        : {os.path.basename(args.image_path)}")
    print(f"  Jumlah fitur DFT  : {X_sample.shape[1]}")
    print(f"  Prob. FAKE (0)    : {fake_prob * 100:.2f} %")
    print(f"  Prob. REAL (1)    : {real_prob * 100:.2f} %")
    print("-" * 60)
    print(f"  Kesimpulan        : {interpret(pred_class)}")
    print("=" * 60)


if __name__ == "__main__":
    main()