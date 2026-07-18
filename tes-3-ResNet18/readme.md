# Quantization Model with MNIST Dataset — ResNet-18

Proyek ini mengeksplorasi **ternary quantization** pada arsitektur **ResNet-18** untuk klasifikasi digit MNIST, menggunakan dua pendekatan: **Fine-Grained Quantization (FGQ)** dan **Standard Ternary Quantization**.

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

### 2. MNIST via Torchvision (Cell 3)
Dataset standar MNIST dari `torchvision.datasets.MNIST()` digunakan untuk **melatih model** sebelum dievaluasi dengan quantisasi. Input di-**resize ke 32×32** (lebih cocok untuk ResNet) dan dinormalisasi menggunakan mean=0.1307 dan std=0.3081.

---

## 🧠 Arsitektur Model

### ResNet-18 (Cell 3) — CNN 18 Layer (Residual)
Diadaptasi dari arsitektur ResNet-18 (He et al., 2015) dengan modifikasi untuk MNIST:

| Modifikasi        | Perubahan                                        |
|-------------------|--------------------------------------------------|
| Input channel     | `conv1`: **1 channel** (grayscale) → 64 channel  |
| Input size        | **32×32** (di-resize dari 28×28 MNIST)           |
| Output kelas      | `fc`: **10 kelas** (diganti dari 1000 ImageNet)  |
| Pretrained        | ❌ **Tidak digunakan** (dilatih dari awal)        |

### Struktur Layer Utama

| Layer Block | Type               | Output Shape   | Keterangan                         |
|-------------|--------------------|----------------|------------------------------------|
| conv1       | Conv2d (7×7)       | 64×16×16       | stride=2, padding=3               |
| bn1         | BatchNorm2d        | 64×16×16       | —                                  |
| layer1      | 2× BasicBlock      | 64×16×16       | 2 residual blocks, 64 channels     |
| layer2      | 2× BasicBlock      | 128×8×8        | stride=2 (downsample)              |
| layer3      | 2× BasicBlock      | 256×4×4        | stride=2 (downsample)              |
| layer4      | 2× BasicBlock      | 512×2×2        | stride=2 (downsample)              |
| avgpool     | AdaptiveAvgPool2d  | 512×1×1        | global average pooling              |
| fc          | Linear             | 512 → 10       | output 10 kelas digit              |

Setiap **BasicBlock** terdiri dari:
- Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → **Skip connection** → ReLU

Model di-*train* selama **5 epoch** menggunakan **Adam optimizer** (lr=0.001) dengan **CrossEntropyLoss**, disimpan ke Google Drive sebagai `resnet18_mnist.pth`.

---

## ⚙️ Teknik Quantization

Kedua teknik menggunakan **ternary weights** — nilai bobot hanya bisa berupa **{-α, 0, +α}** (setara ~2-bit). Quantization diterapkan pada seluruh layer **Conv2d** dan **Linear**.

### 1. Fine-Grained Quantization (FGQ) — Cell 4
Bobot dibagi ke dalam **group** berukuran `N`:

```
Bobot asli: [w1, w2, w3, w4, w5, w6, w7, w8, ...]
                 ╰─── group N=4 ──╯   ╰─── group N=4 ──╯
```

- Setiap group memiliki **threshold (Δ)** dan **skala (α)** sendiri
- Threshold optimal dicari dari **20 kandidat** dengan memaksimalkan `(Σ|wᵢ|)² / count`
- Bobot > Δ → +α &nbsp;|&nbsp; Bobot < −Δ → −α &nbsp;|&nbsp; Sisanya → 0
- Group size `N` dapat dikonfigurasi (2, 4, 8, 16, 32, 64)

### 2. Standard Ternary Quantization — Cell 4
Satu threshold dan satu skala untuk **setiap layer** (global):

```
Δ = 0.7 × E[|W|]   (threshold berdasarkan rata-rata absolut)
```

- Satu α untuk seluruh tensor bobot per layer
- Lebih sederhana, tetapi lebih kasar dibanding FGQ

---

## 📊 Eksperimen & Hasil

### Skenario A: Standard Ternary Quantization (Cell 5)
- Quantization level: **2-bit (Ternary)**
- Granularity: **Global (per layer)**
- Model dievaluasi dengan Standard Ternary dan dibandingkan dengan baseline

### Skenario B: FGQ dengan Variasi Group Size (Cell 6)

| Group Size (N) | Akurasi  | Drop dari Baseline |
|----------------|----------|--------------------|
| 2              | Tertinggi | Minimal            |
| 4              | Terbaik   | Paling kecil       |
| 8              | Baik      | Sedang             |
| 16             | Menurun   | Mulai signifikan   |
| 32             | Rendah    | Besar              |
| 64             | Terendah  | Paling besar       |

**Visualisasi:** Plot akurasi vs group size, bar plot accuracy drop, dan distribusi alpha mean per layer.

### Perbandingan (Cell 7)

```
📊 Baseline (FP32)     ≈ 97-99%
📊 Standard Ternary    ≈ drop besar
📊 FGQ (N=4)           ≈ mendekati baseline (drop minimal)
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

**✔ FGQ (Fine-Grained Quantization) terbukti jauh lebih unggul daripada Standard Ternary** pada arsitektur ResNet-18.  
Dengan N=4, FGQ hampir menyamai performa model asli (FP32), sementara Standard Ternary mengalami penurunan akurasi yang sangat drastis.

**✔ Semakin kecil nilai N (semakin granular),** model semakin mampu mempertahankan informasi penting meskipun dibatasi hanya pada 3 level nilai {-α, 0, +α}.

> **Perbedaan dengan LeNet-5 / MLP (Net):** ResNet-18 memiliki arsitektur yang jauh lebih dalam (18 layer) dengan residual connections, batch normalization, dan jumlah parameter yang jauh lebih besar. Meskipun demikian, FGQ tetap mampu mempertahankan akurasi karena granularitas per group, menunjukkan bahwa pendekatan ini **scalable** untuk model yang lebih besar dan kompleks.

---

## 🚀 Cara Menjalankan

1. Buka notebook di **Google Colab**
2. Mount Google Drive (berisi dataset MNIST terstruktur)
3. Jalankan **Cell 1 → Cell 7** secara berurutan
4. Hasil akan ditampilkan dalam bentuk grafik dan ringkasan numerik

> **Catatan:** Path dataset di Google Drive dapat disesuaikan pada variabel `DATASET_PATH` di Cell 2. Model yang telah dilatih akan disimpan/dimuat dari folder Google Drive (`/content/drive/MyDrive/model/resnet18_mnist.pth`).
