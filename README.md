# 📜 SASKARA – Sistem Transliterasi Aksara Bali Berbasis Deteksi dan Rule-Based

SASKARA merupakan proyek penelitian dan eksperimen yang bertujuan untuk melakukan transliterasi citra Aksara Bali ke teks Latin secara otomatis. Sistem ini dirancang untuk menangani kompleksitas aksara Bali yang bersifat dua dimensi dengan mengombinasikan pendekatan deep learning dan linguistic rule-based.

Pipeline utama terdiri dari:
1. **Pre-processing citra**
2. **Deteksi dan klasifikasi karakter** menggunakan YOLO
3. **Penentuan urutan baca** menggunakan algoritma Major Line
4. **Komposisi aksara ke suku kata Latin** berbasis aturan fonologi
5. **Segmentasi kata** (eksperimental)

Proyek ini dikembangkan sebagai bagian dari laporan percobaan akademik dan ditujukan untuk mendukung digitalisasi serta pelestarian aksara Bali.

---

## ✨ Fitur Utama

- 🔍 **Deteksi karakter Aksara Bali** berbasis YOLO
- 🧭 **Rekonstruksi urutan baca** dengan algoritma Major Line
- 🧩 **Komposisi fonologi aksara Bali ke Latin** berbasis rule-based
- 🖥️ **Antarmuka interaktif** menggunakan Streamlit
- ✂️ **Segmentasi kata Bahasa Bali** (greedy & LSTM – eksperimen)
- 🧪 **Mode debugging spasial** untuk analisis hasil deteksi

---

## 🗂️ Struktur Proyek
"""
.
<br>├── teyvatLontar.py
<br>│ └── Script utama antarmuka Streamlit (UI & pipeline utama).
<br>│
<br>├── preprocess.py
<br>│ └── Modul pre-processing citra (grayscale, median blur, CLAHE, dll.)
<br>│
<br>├── majorLinesAksara.py
<br>│ └── Implementasi algoritma Major Line untuk menentukan baris utama
<br>│ dan urutan pembacaan karakter
<br>│
<br>├── rukeAksara.py
<br>│ └── Script komposisi aksara Bali ke teks Latin berbasis rule-based
<br>│
<br>├── phonologyRulesAksara.yaml
<br>│ └── Konfigurasi aturan fonologi dan spasial aksara Bali
<br>│
<br>├── debugSpatial.py
<br>│ └── Script khusus untuk debugging posisi spasial karakter
<br>│ dan visualisasi relasi antar aksara
<br>│
<br>├── segmentasiKataGreedy.py
<br>│ └── Implementasi segmentasi kata Bahasa Bali menggunakan
<br>│ algoritma greedy longest-match (belum terintegrasi ke Streamlit)
<br>│
<br>├── bahasaBaliDict.csv
<br>│ └── Kamus Bahasa Bali untuk keperluan segmentasi greedy
<br>│
<br>├── testLSTM.py
<br>│ └── Script pengujian model LSTM untuk segmentasi kata
<br>│
<br>├── datasetLSTM/
<br>│ └── vocabulary_clean.json
<br>│ └── Vocabulary hasil preprocessing dataset LSTM
<br>│
<br>└── README.md

---

## ⚙️ Alur Sistem (Pipeline Singkat)

1. **Input citra** aksara Bali
2. **Pre-processing** untuk meningkatkan kualitas citra
3. **YOLO** mendeteksi dan mengklasifikasikan karakter
4. **Major Line Algorithm** menentukan baris utama dan urutan baca
5. **Rule-Based Composition** menyusun aksara menjadi suku kata Latin
6. **(Opsional) Segmentasi kata** untuk membentuk kata utuh

---

## 🧠 Pendekatan yang Digunakan

### Deteksi Karakter
YOLO digunakan untuk mendeteksi dan mengklasifikasikan aksara dasar, sandhangan, dan tanda baca secara langsung dari citra.

### Major Line Algorithm
Menentukan baris utama berdasarkan dominasi aksara wianjana. Digunakan untuk merekonstruksi urutan baca dari hasil object detection.

### Komposisi Rule-Based
Aturan fonologi dan spasial didefinisikan dalam file YAML. Setiap aksara dasar diproses bersama modifier di sekitarnya (gantungan, vokal, tengenan).

### Segmentasi Kata (Eksperimental)
- **Greedy Longest-Match** berbasis kamus Bahasa Bali.
- **LSTM / BiLSTM** untuk eksperimen segmentasi berbasis pembelajaran sekuens.

---

## 🚧 Status Pengembangan

| Komponen | Status |
|----------|---------|
| ✅ Deteksi & komposisi rule-based | Stabil |
| ✅ UI Streamlit | Aktif |
| ⚠️ Segmentasi kata (greedy & LSTM) | Eksperimental |
| 🔧 Integrasi penuh segmentasi ke UI | Belum diimplementasikan |

---

## 📌 Catatan

- Proyek ini bersifat **eksperimental dan akademis**.
- Aturan fonologi dapat diperluas dengan memodifikasi `phonologyRulesAksara.yaml`.
- Performa sangat bergantung pada **kualitas deteksi karakter dari YOLO**.

---

## 📖 Lisensi

Proyek ini dikembangkan untuk keperluan pendidikan dan penelitian. Silakan gunakan, modifikasi, dan kembangkan dengan tetap mencantumkan atribusi.
