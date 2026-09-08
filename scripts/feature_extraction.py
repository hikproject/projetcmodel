#!/usr/bin/env python3
# ============================================================
#  MODUL EKSTRAKSI FITUR DFT (Azimuthal Average)
#  ----------------------------------------------------------
#  Mengikuti metodologi asli dari:
#  Ferreira, S., Antunes, M., & Correia, M. E. (2021).
#  "Exposing Manipulated Photos and Videos in Digital
#  Forensics Analysis", Journal of Imaging, 7(7), 102.
#
#  Fungsi ini digunakan bersama oleh:
#    1. train_benchmark.py  -> training model
#    2. predict_image.py    -> pengujian citra mentah baru
#  sehingga jumlah & makna fitur tetap konsisten (50 fitur).
# ============================================================

import numpy as np
import cv2
import os
import sys

# Modul 'radialProfile' tidak tersedia di PyPI. Kita impor dari file lokal
# (radialProfile.py) yang berada pada folder yang sama dengan script ini.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import radialProfile
from scipy.interpolate import griddata


def extract_features_from_image(img_path, N=50, epsilon=1e-8):
    """
    Mengekstrak N buah fitur DFT (azimuthally averaged power spectrum)
    dari satu citra (grayscale), persis seperti pipeline penulis.

    Parameter
    ----------
    img_path : str
        Lokasi file citra (.jpg / .png / dll).
    N : int
        Jumlah fitur yang dihasilkan (default = 50, sesuai dataset).
    epsilon : float
        Nilai kecil untuk menghindari log(0) pada region real.

    Return
    ------
    numpy.ndarray dengan shape (N,)  -> 1D vektor fitur siap digunakan model.
    """
    # 1) Baca citra dalam mode grayscale (kanal tunggal).
    #    Mode 0 otomatis mengonversi citra warna -> grayscale.
    img = cv2.imread(img_path, 0)

    if img is None:
        raise ValueError(
            f"Gagal membaca citra: {img_path}. "
            "Pastikan file ada dan formatnya didukung OpenCV."
        )

    # 2) Transformasi Fourier 2D.
    #    fft2 : DFT diskrit 2D dari citra.
    #    fftshift : memindahkan komponen frekuensi-0 ke tengah
    #    (DC di tengah), sehingga spektrum simetris radial.
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 3) Koreksi numerik: tambahkan epsilon pada komponen real
    #    (dilakukan penulis pada bagian 'real') agar log tidak meledak.
    fshift += epsilon

    # 4) Magnitude spectrum dalam skala desibel (20*log10).
    #    Mengubah kompleks menjadi nilai riil yang bisa di-plot.
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 5) Azimuthally averaged 1D power spectrum.
    #    Meratakan spektrum 2D menjadi kurva 1D terhadap jari-jari
    #    (jarak dari pusat), menggunakan pustaka radialProfile.
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

    # 6) Interpolasi ke N titik yang seragam.
    #    'points' = koordinat asli dari psd1D (0 .. N-1)
    #    'xi'     = koordinat target sebanyak N (fitur)
    #    Method 'cubic' agar kurva halus (sesuai penulis).
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')

    # 7) Normalisasi terhadap komponen pertama (DC).
    #    Membuat fitur tidak bergantung pada skala kecerahan global citra.
    interpolated = interpolated / interpolated[0]

    return interpolated.astype(np.float64)
