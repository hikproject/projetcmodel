#!/usr/bin/env python3
# ============================================================
#  SCRIPT TRAINING UTAMA & BENCHMARKING
#  ----------------------------------------------------------
#  Tesis S2 - Digital Forensics & Machine Learning
#  "Optimasi Deteksi Manipulasi Citra Multimedia dengan
#   DFT + LightGBM untuk Forensik Digital"
#
#  Menggunakan dataset 50 fitur DFT (.pkl) dari:
#  Ferreira, S., Antunes, M., & Correia, M. E. (2021),
#  Journal of Imaging, 7(7), 102.
#
#  Alur:
#    1. Load data & label dari file .pkl penulis
#    2. 5-Fold Stratified Cross-Validation (kelas seimbang)
#    3. Latih LightGBM Classifier
#    4. Latih SVM Classifier (baseline, parameter persis penulis)
#    5. Hitung Precision, Recall, F1, Accuracy, Confusion Matrix,
#       serta waktu eksekusi pelatihan untuk kedua model
#    6. Simpan model terlatih & hasil evaluasi
# ============================================================

import argparse
import os
import pickle
import time
import warnings

# Tekan warning deprecation dari sklearn (mis. 'probability' SVC) dan
# warning umum lain agar output laporan bersih & mudah dibaca di tesis.
# Nilai default di sini TIDAK mengubah perilaku ML.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import lightgbm as lgb  # Pustaka Gradient Boosting (target utama penelitian)

import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif (tidak butuh display GUI)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#  KONSTANTA
# ============================================================
LABEL_NAMES = {0: "Fake (Manipulated)", 1: "Real (Authentic)"}


# ============================================================
#  1) FUNGSI MEMUAT DATASET DARI .pkl
# ============================================================
def load_dataset(pkl_path):
    """
    Membaca file .pkl penulis yang berisi dictionary dua kunci:
        data["data"]  -> X (matriks fitur 50 DFT)
        data["label"] -> y (0 = fake, 1 = real)

    Return: (X, y) sebagai numpy array.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = np.asarray(data["data"], dtype=np.float64)
    y = np.asarray(data["label"], dtype=np.int32).ravel()

    print("=" * 70)
    print("Dataset berhasil dimuat:")
    print(f"  File        : {os.path.basename(pkl_path)}")
    print(f"  Jumlah baris: {X.shape[0]}")
    print(f"  Jumlah fitur: {X.shape[1]}")
    print(f"  Kelas 0 (Fake): {int((y == 0).sum())}  |  Kelas 1 (Real): {int((y == 1).sum())}")
    print("=" * 70)
    return X, y


# ============================================================
#  2) FUNGSI TRAINING SATU SPLIT + EVALUASI
# ============================================================
def run_fold(model, X_train, y_train, X_test, y_test):
    """
    Melatih satu model pada satu fold dan mengembalikan metrik evaluasi.
    Waktu pelatihan diukur terpisah sehingga mencerminkan biaya training
    (bukan termasuk waktu prediksi / predict_proba yang bisa lambat utk SVM).
    """
    start = time.time()
    model.fit(X_train, y_train)          # Hanya mengukur pelatihan
    train_time = time.time() - start

    y_pred = model.predict(X_test)       # Prediksi label kelas

    metrics = {
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "train_time": train_time,
        "y_pred":     y_pred,            # disimpan utk agregasi confusion matrix
        "y_test":     y_test,
    }
    return metrics


# ============================================================
#  3) FUNGSI PENYIMPANAN MODEL
# ============================================================
def save_models(lgb_model, svm_model, output_dir):
    """Menyimpan model terlatih ke folder models/."""
    os.makedirs(output_dir, exist_ok=True)

    # LightGBM -> format native teks (portable, tanpa pickle)
    lgb_path = os.path.join(output_dir, "lightgbm_model.txt")
    lgb_model.booster_.save_model(lgb_path)
    print(f"[SAVED] LightGBM model -> {lgb_path}")

    # SVM -> pickle (tidak ada format native)
    svm_path = os.path.join(output_dir, "svm_model.pkl")
    with open(svm_path, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"[SAVED] SVM model -> {svm_path}")


# ============================================================
#  4) FUNGSI VISUALISASI HASIL
# ============================================================
def plot_results(conf_matrix, model_names, accuracies, f1s, output_dir):
    """Membuat 2 plot: confusion matrix gabungan & perbandingan akurasi/F1."""
    os.makedirs(output_dir, exist_ok=True)

    # 4a. Confusion Matrix gabungan (semua fold) untuk kedua model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, cm) in zip(axes, zip(model_names, conf_matrix)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Fake", "Real"],
                    yticklabels=["Fake", "Real"])
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] Confusion Matrix plot -> {cm_path}")

    # 4b. Perbandingan Akurasi & F1-Score antar model
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, f1s, width, label="F1-Score", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Akurasi & F1-Score (Rata-rata 5-Fold)")
    ax.legend()
    fig.tight_layout()
    bar_path = os.path.join(output_dir, "accuracy_f1_comparison.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] Bar comparison plot -> {bar_path}")


# ============================================================
#  5) FUNGSI UTAMA
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Training & benchmarking LightGBM vs SVM pada dataset 50 fitur DFT."
    )
    parser.add_argument(
        "pkl_path",
        help="Path ke file .pkl dataset (mis. datasets/train_photos.pkl).",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Jumlah fold cross-validation (default: 5).",
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Folder untuk menyimpan model terlatih (default: models).",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Folder untuk hasil evaluasi & plot (default: output).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Seed untuk reproduktibilitas (default: 42).",
    )
    args = parser.parse_args()

    # --- 1. Load dataset ---
    X, y = load_dataset(args.pkl_path)

    # --- 2. Setup Stratified K-Fold ---
    #    'Stratified' menjaga proporsi kelas yang sama di tiap fold
    #    (penting karena dataset berimbang fake vs real).
    #    shuffle=True + random_state agar urutan acak & dapat direproduksi.
    skf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state,
    )

    # --- 3. Inisialisasi model ---
    #    LightGBM: Gradient Boosting (target penelitian).
    #    n_jobs=-1 memakai semua core (i3-1215U: 8 thread).
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=args.random_state,
        verbose=-1,
    )

    #    SVM baseline: parameter PERSIS seperti yang dipakai penulis
    #    (Ferreira et al., 2021) agar perbandingan fair.
    svm_model = SVC(
        C=6.37,
        kernel='rbf',
        gamma=0.86,
        probability=True,   # dipakai hanya untuk konsistensi dgn penulis
        random_state=args.random_state,
    )

    # --- 4. Loop K-Fold untuk kedua model ---
    lgb_metrics, svm_metrics = [], []
    lgb_cm_total = np.zeros((2, 2), dtype=int)   # akumulasi confusion matrix
    svm_cm_total = np.zeros((2, 2), dtype=int)

    print("\nMemulai 5-Fold Stratified Cross-Validation...\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"--- Fold {fold}/{args.folds} ---")

        # LightGBM
        m_lgb = run_fold(lgb_model, X_train, y_train, X_test, y_test)
        lgb_metrics.append(m_lgb)
        lgb_cm_total += confusion_matrix(y_test, m_lgb["y_pred"])

        # SVM
        m_svm = run_fold(svm_model, X_train, y_train, X_test, y_test)
        svm_metrics.append(m_svm)
        svm_cm_total += confusion_matrix(y_test, m_svm["y_pred"])

        print(f"  LightGBM acc={m_lgb['accuracy']:.4f}  f1={m_lgb['f1']:.4f}  "
              f"train_time={m_lgb['train_time']:.2f}s")
        print(f"  SVM      acc={m_svm['accuracy']:.4f}  f1={m_svm['f1']:.4f}  "
              f"train_time={m_svm['train_time']:.2f}s")
        print()

    # --- 5. Agregasi hasil lintas fold ---
    def summarize(metrics_list, cm_total, model_name):
        df = pd.DataFrame(metrics_list)
        agg = {
            "Model": model_name,
            "Accuracy (mean+/-std)":  f"{df['accuracy'].mean():.4f} +/- {df['accuracy'].std():.4f}",
            "Precision (mean)":       f"{df['precision'].mean():.4f}",
            "Recall (mean)":          f"{df['recall'].mean():.4f}",
            "F1-Score (mean)":        f"{df['f1'].mean():.4f}",
            "Train Time (s, mean)":   f"{df['train_time'].mean():.2f}",
        }
        return agg, df, cm_total

    lgb_agg, lgb_df, lgb_cm = summarize(lgb_metrics, lgb_cm_total, "LightGBM")
    svm_agg, svm_df, svm_cm = summarize(svm_metrics, svm_cm_total, "SVM")

    # --- 6. Cetak laporan akhir ---
    print("=" * 70)
    print("              HASIL EVALUASI (STRATIFIED 5-FOLD CV)")
    print("=" * 70)
    summary = pd.DataFrame([lgb_agg, svm_agg])
    print(summary.to_string(index=False))

    print("\n--- Confusion Matrix (akumulasi semua fold) ---")
    for name, cm in [("LightGBM", lgb_cm), ("SVM", svm_cm)]:
        print(f"\n{name}:")
        print(pd.DataFrame(cm, index=["Actual Fake", "Actual Real"],
                           columns=["Pred Fake", "Pred Real"]))
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Positives (Real benar) : {tp}")
        print(f"  False Negatives (Fake dianggap Real): {fn}")
        print(f"  False Positives (Real dianggap Fake): {fp}")
        print(f"  True Negatives (Fake benar) : {tn}")

    # --- 7. Simpan laporan ke file teks ---
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "results.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  HASIL EVALUASI - DFT + LightGBM vs SVM\n")
        f.write(f"  Dataset: {args.pkl_path}\n")
        f.write(f"  Folds:   {args.folds}\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary.to_string(index=False) + "\n\n")
        f.write("--- Confusion Matrix ---\n")
        for name, cm in [("LightGBM", lgb_cm), ("SVM", svm_cm)]:
            f.write(f"\n{name}:\n")
            f.write(pd.DataFrame(cm, index=["Actual Fake", "Actual Real"],
                                 columns=["Pred Fake", "Pred Real"]).to_string() + "\n")
        f.write("\n--- Classification Report (LightGBM) ---\n")
        f.write(classification_report(
            np.concatenate([m["y_test"] for m in lgb_metrics]),
            np.concatenate([m["y_pred"] for m in lgb_metrics]),
            digits=4))
        f.write("\n--- Classification Report (SVM) ---\n")
        f.write(classification_report(
            np.concatenate([m["y_test"] for m in svm_metrics]),
            np.concatenate([m["y_pred"] for m in svm_metrics]),
            digits=4))
    print(f"\n[SAVED] Laporan hasil -> {report_path}")

    # --- 8. Simpan model (dilatih ulang pada SEMUA data untuk generalisasi) ---
    #    Model final dilatih penuh pada seluruh dataset agar siap dipakai
    #    memprediksi citra baru (bukan hanya pada subset fold).
    final_lgb = lgb.LGBMClassifier(**lgb_model.get_params())
    final_svm = SVC(**svm_model.get_params())
    final_lgb.fit(X, y)
    final_svm.fit(X, y)
    save_models(final_lgb, final_svm, args.model_dir)

    # --- 9. Plot visualisasi ---
    plot_results([lgb_cm, svm_cm],
                 ["LightGBM", "SVM"],
                 [lgb_df["accuracy"].mean(), svm_df["accuracy"].mean()],
                 [lgb_df["f1"].mean(), svm_df["f1"].mean()],
                 args.output_dir)

    print("\nSelesai. Silakan cek folder 'output/' dan 'models/'.")


if __name__ == "__main__":
    main()
