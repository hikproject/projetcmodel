# ProjetCModel — Deteksi Manipulasi Citra dengan DFT + LightGBM

Jurnal S2 Magister Teknik Informatika — *Digital Forensics & Machine Learning*.

> **Judul:** Optimasi Deteksi Manipulasi Citra Multimedia Menggunakan Kombinasi
> Discrete Fourier Transform (DFT) dan Algoritma Gradient Boosting (LightGBM)
> untuk Forensik Digital.
>
> Pengembangan dari: Ferreira, S., Antunes, M., & Correia, M. E. (2021),
> *Exposing Manipulated Photos and Videos in Digital Forensics Analysis*,
> Journal of Imaging, 7(7), 102. (baseline: SVM)

---

## 1) Setup Environment (Linux / Arch / Omarchy)

Sistem yang digunakan memiliki **Python 3.14** bawaan tanpa conda/pip/uv global.
Kita buat virtual environment (`venv`) lalu instal dependensi dari `requirements.txt`.

```bash
# Masuk ke folder proyek

# 1. Buat virtual environment
python3 -m venv .venv

# 2. Aktivasi (ganti sesuai shell Anda)
source .venv/bin/activate            # bash / zsh
# source .venv/bin/activate.fish     # jika memakai fish shell

# 3. Upgrade pip (pastikan stabil)
pip install --upgrade pip

# 4. Instal semua dependensi
pip install -r requirements.txt
```

### Troubleshooting instalasi di Python 3.14

Python 3.14 sangat baru; beberapa paket (terutama **LightGBM**) mungkin belum
memiliki *wheel* binary dan memerlukan kompilasi dari sumber.

```bash
# Jika instalasi lightgbm gagal, instal toolchain build lalu kompilasi ulang:
sudo pacman -S cmake gcc make        # Arch/Omarchy
pip install lightgbm --no-binary lightgbm

# Jika 'radialProfile' gagal ditemukan, pastikan nama paket sesuai PyPI:
pip install radialProfile
```

> **Catatan:** Jika `lightgbm` tetap tidak bisa dikompilasi di 3.14, opsi paling
> stabil adalah menginstal Python 3.11 (yang punya wheel lengkap) — misal via
> `mise`/`pyenv` — lalu ulangi langkah `venv` di atas di atas Python 3.11.

---

## 2) Struktur Folder

```
projetcmodel/
├── .venv/                    # Virtual environment (tidak di-commit)
├── requirements.txt          # Dependensi + versi
├── README.md                 # Dokumen ini
├── scripts/
│   ├── feature_extraction.py # Modul ekstraksi 50 fitur DFT (dipakai bersama)
│   ├── train_benchmark.py    # Training utama: LightGBM vs SVM (5-Fold CV)
│   └── predict_image.py      # Pengujian citra mentah baru (out-of-sample)
├── data/                     # (opsional) salin/symlink dataset .pkl
├── models/                   # Model terlatih tersimpan
│   ├── lightgbm_model.txt
│   └── svm_model.pkl
└── output/                   # Hasil evaluasi & plot
    ├── results.txt
    ├── confusion_matrices.png
    └── accuracy_f1_comparison.png
```

**Dataset penulis** berada di luar folder proyek ini (repo kloning):
`/home/admin/Projects/tesis/Photos-Videos-Manipulations-Dataset-main/datasets/`.

Format `.pkl` penulis adalah dictionary dengan kunci:
- `data["data"]` → `X` (matriks fitur 50 DFT)
- `data["label"]` → `y` (**0 = fake**, **1 = real**)

Tersedia:
- `train_photos.pkl` − hanya foto
- `train_videos.pkl` − hanya frame video
- `train_all.pkl` − foto + video

---

## 3) Menjalankan Training & Benchmarking

```bash
source .venv/bin/activate

# Contoh: training pada dataset foto
python scripts/train_benchmark.py \
    /home/admin/Projects/tesis/Photos-Videos-Manipulations-Dataset-main/datasets/train_photos.pkl

# Contoh lain: dataset gabungan foto+video, dengan 10 fold
python scripts/train_benchmark.py \
    /home/admin/Projects/tesis/Photos-Videos-Manipulations-Dataset-main/datasets/train_all.pkl \
    --folds 10
```

**Yang dilakukan script ini:**
1. Memuat data & label dari `.pkl`
2. **5-Fold Stratified Cross-Validation** (kelas seimbang tiap fold)
3. Melatih **LightGBM** (gradient boosting, `n_jobs=-1` → pakai semua 8 thread)
4. Melatih **SVM baseline** (`C=6.37, kernel='rbf', gamma=0.86` — persis penulis)
5. Menghitung rata-rata **Accuracy, Precision, Recall, F1-Score, Confusion Matrix**
6. Mengukur **waktu pelatihan** (detik) untuk kedua model
7. Menyimpan model final ke `models/` dan laporan/plot ke `output/`

---

## 4) Pengujian Citra Mentah Baru (Out-of-Sample)

Membaca **satu citra** (`.jpg`/`.png`), ekstrak 50 fitur DFT, lalu prediksi keasliannya:

```bash
source .venv/bin/activate

python scripts/predict_image.py /path/ke/gambar.jpg
# atau tentukan model eksplisit:
python scripts/predict_image.py /path/ke/gambar.png --model models/lightgbm_model.txt
```

**Output** menampilkan probabilitas `Fake (0)` vs `Real (1)` serta kesimpulan forensik.
Normalisasi & ekstraksi fitur **identik** dengan proses saat training
(`feature_extraction.py`), sehingga hasil konsisten.
