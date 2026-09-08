"""
radialProfile.py
=================
Modul kecil untuk menghitung *azimuthally averaged* 1D power spectrum.

Implementasi ini adalah implementasi asli yang dipakai oleh penulis dataset
(Ferreira et al., 2021), diambil dari AstroBetter / proyek open-source
(https://www.astrobetter.com/wiki/python_radial_profiles).

Catatan: paket ini TIDAK tersedia di PyPI sebagai `pip install radialProfile`.
Penulis aslinya menyertakan file `radialProfile.py` ini langsung di dalam
foldernya, maka di sini file juga disertakan sebagai modul lokal agar
metodologi ekstraksi fitur ter-replikasi secara persis.
"""

import numpy as np


def azimuthalAverage(image, center=None):
    """
    Menghitung radial profile rata-rata azimuthal dari citra 2D.

    Parameter
    ----------
    image  : array 2D (spektrum magnitude) yang akan diratakan.
    center : koordinat [x, y] pusat. Jika None, memakai pusat citra
             (termasuk piksel pecahan).

    Return
    ------
    radial_prof : array 1D berisi rata-rata per bin radius (jarak dari pusat).
                  Panjangnya = jumlah jari-jari integer yang terwakili.
    """
    # Buat koordinat indeks piksel dari citra.
    y, x = np.indices(image.shape)

    # Jika pusat tidak diberikan, hitung pusat citra (bisa pecahan).
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    # Jarak radial setiap piksel dari pusat (hipotenusa euclidean).
    r = np.hypot(x - center[0], y - center[1])

    # Urutkan piksel berdasarkan jarak radial.
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Ambil bagian integer dari jari-jari (ukuran bin = 1).
    r_int = r_sorted.astype(int)

    # Temukan posisi di mana radius berubah -> batas tiap bin radial.
    deltar = r_int[1:] - r_int[:-1]   # 0 bila radius sama, !=0 saat ganti bin
    rind = np.where(deltar)[0]        # lokasi radius berubah
    nr = rind[1:] - rind[:-1]         # jumlah piksel tiap bin radius

    # Cumulative sum untuk menghitung total nilai tiap bin radius.
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    # Rata-rata per bin = total / jumlah piksel pada bin tsb.
    radial_prof = tbin / nr

    return radial_prof
