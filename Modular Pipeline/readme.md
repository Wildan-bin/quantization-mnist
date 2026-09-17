# Modular Pipeline — Quantization MNIST & Fashion MNIST

## 📌 Gambaran Umum

Folder ini berisi **dua notebook modular** yang menerapkan **pendekatan modular** untuk eksperimen kuantisasi model *ternary* dan *fine-grained ternary quantization (FGQ)* — termasuk varian **asymmetric threshold** — pada dua dataset:

| Notebook | Dataset | Arsitektur |
|---|---|---|
| `Quantization_Fashion_MNIST.ipynb` | **Fashion-MNIST** (normalisasi 0.5 / 0.5) | LeNet-5 & ResNet-18 |
| `Quantization_MNIST.ipynb` *(baru)* | **MNIST** (normalisasi 0.1307 / 0.3081) | LeNet-5 & ResNet-18 |

Kedua notebook berbagi **blok fungsi modular yang identik**; perbedaannya hanya terletak pada fungsi setup dataset dan checkpoint model.

Pendekatan ini merupakan hasil refaktorisasi dari skema monolitik sebelumnya, memisahkan setiap tanggung jawab ke dalam fungsi-fungsi independen yang dapat digunakan kembali.

---

## 🧩 Skema Modular

Pipeline dibagi menjadi **blok fungsi modular** dan **blok eksekusi**:

### A. Blok Fungsi Modular (Definisi)

| Fungsi | Tanggung Jawab |
|---|---|
| `setup_mnist_environment()` | Mount Google Drive, inisialisasi path, siapkan DataLoader **MNIST** |
| `setup_fashion_mnist_environment()` | Mount Google Drive, inisialisasi path, siapkan DataLoader **Fashion-MNIST** |
| `load_and_prepare_model()` | Load model dari checkpoint (jika ada) **atau** training dari awal + simpan + riwayat loss/accuracy |
| `evaluate_accuracy()` | Evaluasi akurasi top-1 pada data loader |
| `standard_ternary_quantize()` | Kuantisasi ternary standar (threshold `Δ = 0.7 × mean(\|W\|)`) per tensor |
| `apply_standard_ternary_to_model()` | Terapkan ternary standar ke seluruh layer Conv2d/Linear via `copy.deepcopy()` |
| `ternary_quantize_group()` | Kuantisasi ternary per grup (FGQ) — mode **symmetric** (optimasi 20 kandidat threshold) **atau asymmetric** (threshold & alpha terpisah untuk nilai positif/negatif) |
| `apply_fgq_to_model()` | Terapkan FGQ ke seluruh layer dengan konfigurasi `group_size` & `asymmetric` |
| `find_best_fgq_model()` | Evaluasi semua group size, kembalikan yang akurasinya terbaik |
| `LeNet5` / `ResNet18` (class) | Definisi arsitektur model terpusat |

### B. Blok Eksekusi

| Tahap | Aktivitas |
|---|---|
| Setup Environment | Panggil `setup_mnist_environment()` / `setup_fashion_mnist_environment()` + `load_and_prepare_model()` untuk tiap arsitektur |
| Standard Ternary | Panggil `apply_standard_ternary_to_model()` + `evaluate_accuracy()` |
| FGQ (N = 4, 8, 16) | Panggil `apply_fgq_to_model(..., asymmetric=True)` dengan berbagai group size |
| Perbandingan Akurasi | Tampilkan perbandingan akurasi semua skema per model |

---

## ⚖️ Fitur Baru: Asymmetric Threshold pada FGQ

Sebelumnya, FGQ hanya menggunakan threshold **symmetric** — satu `delta` dan satu `alpha` yang sama untuk nilai positif & negatif dalam satu grup. Kini `ternary_quantize_group()` mendukung mode **asymmetric** (`asymmetric=True`), dengan logika:

- **Delta & alpha positif dan negatif dihitung terpisah** (`d_pos`/`a_pos` vs `d_neg`/`a_neg`)
- Threshold positif: `d_pos = 0.7 × mean(pos_vals)`, lalu `a_pos = mean(pos_vals[pos_vals > d_pos])`
- Threshold negatif: `d_neg = 0.7 × mean(|neg_vals|)`, lalu `a_neg = mean(neg_vals[neg_vals < -d_neg])`
- Bobot yang tidak melewati threshold masing-masing di-nol-kan

```python
# FGQ symmetric (default): satu threshold per grup
model_fgq_sym, _ = apply_fgq_to_model(model_fp32, group_size=4, asymmetric=False)

# FGQ asymmetric: threshold & alpha terpisah untuk nilai + dan -
model_fgq_asym, _ = apply_fgq_to_model(model_fp32, group_size=4, asymmetric=True)
```

Dalam eksperimen, FGQ dijalankan dengan **`asymmetric=True`** untuk group size `{4, 8, 16}` pada kedua arsitektur.

---

## 🆚 Perbedaan dengan Skema Lama

| Aspek | Skema Lama (Monolitik) | Skema Baru (Modular) |
|---|---|---|
| **Struktur Notebook** | Satu notebook besar dengan semua kode inline dan berulang | Fungsi modular terdefinisi di awal, eksekusi terpisah di akhir |
| **Dataset** | *Custom dataset* dari Google Drive (`HandwrittenDigitDataset`) — folder gambar per kelas | Dataset standar **torchvision (MNIST & Fashion-MNIST)** — *download & go* |
| **Model** | Model (SimpleCNN/LeNet5) didefinisikan berkali-kali di berbagai cell | Definisi class `LeNet5` & `ResNet18` di cell terpusat |
| **Manajemen Checkpoint** | Simpan/load `state_dict` saja, tanpa riwayat training | Simpan `state_dict` + `history` (loss & accuracy per epoch) |
| **Keamanan Model Asli** | Tidak ada salinan otomatis — kuantisasi langsung memodifikasi model | `copy.deepcopy()` — model asli (FP32) tetap aman |
| **Evaluasi** | Fungsi `evaluate_accuracy()` didefinisikan ulang di beberapa cell | Satu fungsi global, dipanggil berkali-kali |
| **Kuantisasi Ternary** | Hanya FGQ (`ternary_quantize_group`) yang diimplementasikan | **Standard Ternary** + **FGQ** dengan mode **symmetric** & **asymmetric threshold** |
| **Eksperimen Group Size** | Perbandingan N = 2, 4, 8, 16, 32, 64 dengan loop eksplisit | Panggil fungsi dengan parameter `group_size` & `asymmetric` — satu baris per skenario |
| **Fine-Tuning** | Embed re-ternarization di tiap langkah optimizer | Belum diimplementasikan (mudah ditambahkan via fungsi terpisah) |
| **Reusability** | Sulit — kode terkait erat dengan dataset/model tertentu | Tinggi — fungsi dapat digunakan untuk dataset/model lain dengan perubahan minimal |
| **Duplikasi Kode** | Tinggi — definisi model dan utilitas diulang antar section/notebook | Rendah — setiap fungsi didefinisikan sekali |

### Contoh Perbedaan Kode

**Skema Lama** — Dataset loading manual dari Drive:
```python
class HandwrittenDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # ~50 baris untuk membaca folder dan file gambar
        ...
```

**Skema Baru** — Satu baris:
```python
train_set = datasets.FashionMNIST(root='./data_fashion', train=True,
                                   download=True, transform=transform)
```

**Skema Lama** — Kuantisasi & evaluasi inline (duplikasi kode):
```python
model, alpha_means, _ = apply_fgq_to_model(model, group_size=N)
acc = evaluate_accuracy(model, test_loader)
```

**Skema Baru** — Panggil fungsi modular + baseline aman (+ asymmetric threshold):
```python
model_fgq4, mean_alphas4 = apply_fgq_to_model(model_fp32, group_size=4, asymmetric=True)
acc_fgq4 = evaluate_accuracy(model_fgq4, test_loader, device)
```

---

## 🧪 Hasil Eksperimen (Contoh)

Setelah menjalankan notebook, hasil perbandingan akurasi akan tampil untuk **tiap arsitektur** (LeNet-5 & ResNet-18) seperti berikut:

| Skema | LeNet-5 | ResNet-18 |
|---|---|---|
| Baseline FP32 | ~99% (MNIST) / ~91% (Fashion-MNIST) | ~99% (MNIST) / ~91% (Fashion-MNIST) |
| Standard Ternary | ~X% | ~X% |
| FGQ N=4 (asymmetric) | ~X% | ~X% |
| FGQ N=8 (asymmetric) | ~X% | ~X% |
| FGQ N=16 (asymmetric) | ~X% | ~X% |

> **Catatan:** Angka aktual bergantung pada riwayat training model di session tersebut. FGQ dijalankan dengan `asymmetric=True`.

---

## 🚀 Cara Menggunakan

```python
# 1. Setup environment (pilih sesuai notebook / dataset)
model_path, train_loader, test_loader = setup_mnist_environment('lenet5_mnist_baseline.pth')
# atau: model_path, ... = setup_fashion_mnist_environment('lenet5_fashion_baseline.pth')

# 2. Load / Train model baseline
model_fp32, history = load_and_prepare_model(LeNet5, model_path,
                                              train_loader, test_loader, device)

# 3. Standard Ternary Quantization
model_ternary, alphas = apply_standard_ternary_to_model(model_fp32)
acc_ternary = evaluate_accuracy(model_ternary, test_loader, device)

# 4. Fine-Grained Ternary Quantization (FGQ) — asymmetric threshold
model_fgq, alphas = apply_fgq_to_model(model_fp32, group_size=4, asymmetric=True)
acc_fgq = evaluate_accuracy(model_fgq, test_loader, device)

# 5. Bandingkan
print(f"Baseline: {history['accuracy'][-1]:.2f}%")
print(f"Ternary:  {acc_ternary:.2f}%")
print(f"FGQ N=4 (asym):  {acc_fgq:.2f}%")
```

---

## 📂 Struktur Folder

```
Modular Pipeline/
├── Quantization_Fashion_MNIST.ipynb   # Notebook modular — dataset Fashion-MNIST
├── Quantization_MNIST.ipynb           # Notebook modular (baru) — dataset MNIST
└── readme.md                          # File ini
```

Notebook lain di folder induk (`tes-1-Net/`, `tes-2-LeNet5/`, `tes-3-ResNet18/`) merupakan eksperimen dengan skema lama (monolitik) untuk berbagai dataset dan arsitektur.
