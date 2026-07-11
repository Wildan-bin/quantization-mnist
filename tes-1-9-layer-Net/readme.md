# Quantization Model with MNIST Dataset

Proyek ini mengeksplorasi **ternary quantization** pada model neural network untuk klasifikasi digit MNIST, menggunakan dua pendekatan: **Fine-Grained Quantization (FGQ)** dan **Standard Ternary Quantization**.

---

## 📂 Dataset

### 1. Custom MNIST dari Google Drive (Cell 2)
Dataset handwritten digit (0–9) disimpan di Google Drive dengan struktur folder:

```
MNIST/
├── training/
│   ├── 0/  (gambar angka 0)
│   ├── 1/  (gambar angka 1)
│   └── .../
└── testing/
    ├── 0/
    ├── 1/
    └── .../
```

Kelas `HandwrittenDigitDataset` membaca gambar dari folder, meresize ke **28×28 pixels (grayscale)**, dan menerapkan normalisasi ke rentang **[-1, 1]**.

### 2. MNIST via Torchvision (Cell 4)
Dataset standar MNIST dari `torchvision.datasets.MNIST()` digunakan untuk **melatih model** sebelum dievaluasi dengan quantisasi. Normalisasi menggunakan mean=0.1307 dan std=0.3081.

---

## 🧠 Arsitektur Model

### Net (Cell 4) — MLP 3 Layer
Digunakan sebagai model utama untuk eksperimen quantisasi:

| Layer | Shape             | Aktivasi |
|-------|-------------------|----------|
| fc1   | 784 → 128         | ReLU     |
| fc2   | 128 → 64          | ReLU     |
| fc3   | 64 → 10           | —        |

Model di-*train* selama **5 epoch** menggunakan **Adam optimizer** (lr=0.001) dengan **CrossEntropyLoss**, disimpan ke Google Drive sebagai `final_model.pth`.

---

## ⚙️ Teknik Quantization

Kedua teknik menggunakan **ternary weights** — nilai bobot hanya bisa berupa **{-α, 0, +α}** (setara ~2-bit).

### 1. Fine-Grained Quantization (FGQ) — Cell 5
Bobot dibagi ke dalam **group** berukuran `N`:

```
Bobot asli: [w1, w2, w3, w4, w5, w6, w7, w8, ...]
                 ╰─── group N=4 ──╯   ╰─── group N=4 ──╯
```

- Setiap group memiliki **threshold (Δ)** dan **skala (α)** sendiri
- Threshold optimal dicari dari **20 kandidat** dengan memaksimalkan `(Σ|wᵢ|)² / count`
- Bobot > Δ → +α &nbsp;|&nbsp; Bobot < −Δ → −α &nbsp;|&nbsp; Sisanya → 0
- Group size `N` dapat dikonfigurasi (2, 4, 8, 16, 32, 64)

### 2. Standard Ternary Quantization — Cell 5.1
Satu threshold dan satu skala untuk **setiap layer** (global):

```
Δ = 0.7 × E[|W|]   (threshold berdasarkan rata-rata absolut)
```

- Satu α untuk seluruh tensor bobot per layer
- Lebih sederhana, tetapi lebih kasar dibanding FGQ

---

## 📊 Eksperimen & Hasil

### Skenario A: FGQ dengan Variasi Group Size (Cell 6)

| Group Size (N) | Akurasi  | Drop dari Baseline |
|----------------|----------|--------------------|
| 2              | Tertinggi | Minimal            |
| 4              | Terbaik   | Paling kecil       |
| 8              | Baik      | Sedang             |
| 16             | Menurun   | Mulai signifikan   |
| 32             | Rendah    | Besar              |
| 64             | Terendah  | Paling besar       |

**Visiualisasi:** Plot akurasi vs group size, bar plot accuracy drop, dan distribusi alpha mean per layer.

### Skenario B: Standard Ternary Quantization (Cell 7)
- Quantization level: **2-bit (Ternary)**
- Granularity: **Global (per layer)**
- Hasil: Akurasi turun **sangat drastis** dari baseline

### Perbandingan (Cell 8)

```
📊 Baseline (FP32)     ≈ 97-98%
📊 FGQ (N=4)           ≈ mendekati baseline (drop minimal)
📊 Standard Ternary    ≈ drop sangat besar
```

---

## 💡 Kesimpulan Utama

| Aspek                     | FGQ (Fine-Grained)                          | Standard Ternary                |
|---------------------------|---------------------------------------------|---------------------------------|
| Granularitas              | **Per group** (N bobot/group)              | **Global** (per layer)          |
| Preservasi Informasi      | ✅ **Sangat baik** (N kecil)               | ❌ **Buruk**                    |
| Group Size Optimal        | **N = 4**                                   | —                               |
| Akurasi Terbaik           | ✅ **Hampir menyamai FP32**                | ❌ **Jauh di bawah baseline**   |
| Kompleksitas Implementasi | Sedang                                       | Sederhana                       |

**✔ FGQ (Fine-Grained Quantization) terbukti jauh lebih unggul daripada Standard Ternary.**  
Dengan N=4, FGQ hampir menyamai performa model asli (FP32), sementara Standard Ternary mengalami penurunan akurasi yang sangat drastis.

**✔ Semakin kecil nilai N (semakin granular),** model semakin mampu mempertahankan informasi penting meskipun dibatasi hanya pada 3 level nilai {-α, 0, +α}.

---

## 🚀 Cara Menjalankan

1. Buka notebook di **Google Colab**
2. Mount Google Drive (berisi dataset MNIST terstruktur)
3. Jalankan **Cell 1 → Cell 8** secara berurutan
4. Hasil akan ditampilkan dalam bentuk grafik dan ringkasan numerik

> **Catatan:** Path dataset di Google Drive dapat disesuaikan pada variabel `DATASET_PATH` di Cell 2.
