# Modular Pipeline — Quantization Fashion MNIST

## 📌 Gambaran Umum

Notebook `Quantization_Fashion_MNIST.ipynb` menerapkan **pendekatan modular** untuk eksperimen kuantisasi model *ternary* dan *fine-grained ternary quantization (FGQ)* pada dataset **Fashion-MNIST** dengan arsitektur **LeNet-5**.

Pendekatan ini merupakan hasil refaktorisasi dari skema monolitik sebelumnya, memisahkan setiap tanggung jawab ke dalam fungsi-fungsi independen yang dapat digunakan kembali.

---

## 🧩 Skema Modular (Baru)

Pipeline dibagi menjadi **blok fungsi modular** dan **blok eksekusi**:

### A. Blok Fungsi Modular (Definisi)

| Fungsi | Tanggung Jawab |
|---|---|
| `setup_fashion_mnist_environment()` | Mount Google Drive, inisialisasi path penyimpanan, siapkan DataLoader Fashion-MNIST |
| `load_and_prepare_model()` | Load model dari checkpoint (jika ada) **atau** training dari awal + simpan + riwayat loss/accuracy |
| `evaluate_accuracy()` | Evaluasi akurasi top-1 pada data loader |
| `standard_ternary_quantize()` | Kuantisasi ternary standar (threshold `Δ = 0.7 × mean(\|W\|)`) per tensor |
| `apply_standard_ternary_to_model()` | Terapkan ternary standar ke seluruh layer Conv2d/Linear via `copy.deepcopy()` |
| `ternary_quantize_group()` | Kuantisasi ternary per grup (FGQ) dengan optimasi threshold (20 kandidat) |
| `apply_fgq_to_model()` | Terapkan FGQ ke seluruh layer dengan konfigurasi `group_size` |
| `LeNet5` (class) | Definisi arsitektur model terpusat |

### B. Blok Eksekusi

| Tahap | Aktivitas |
|---|---|
| Setup Environment | Panggil `setup_fashion_mnist_environment()` + `load_and_prepare_model()` |
| Standard Ternary | Panggil `apply_standard_ternary_to_model()` + `evaluate_accuracy()` |
| FGQ (N = 4, 8, 16) | Panggil `apply_fgq_to_model()` dengan berbagai group size |
| Perbandingan Akurasi | Tampilkan tabel perbandingan semua skema |

---

## 🆚 Perbedaan dengan Skema Lama

| Aspek | Skema Lama (Monolitik) | Skema Baru (Modular) |
|---|---|---|
| **Struktur Notebook** | Satu notebook besar dengan semua kode inline dan berulang | Fungsi modular terdefinisi di awal, eksekusi terpisah di akhir |
| **Dataset** | *Custom dataset* dari Google Drive (`HandwrittenDigitDataset`) — folder gambar per kelas | Dataset standar **torchvision (Fashion-MNIST)** — *download & go* |
| **Model** | Model (SimpleCNN/LeNet5) didefinisikan berkali-kali di berbagai cell | Satu definisi class `LeNet5` di cell terpusat |
| **Manajemen Checkpoint** | Simpan/load `state_dict` saja, tanpa riwayat training | Simpan `state_dict` + `history` (loss & accuracy per epoch) |
| **Keamanan Model Asli** | Tidak ada salinan otomatis — kuantisasi langsung memodifikasi model | `copy.deepcopy()` — model asli (FP32) tetap aman |
| **Evaluasi** | Fungsi `evaluate_accuracy()` didefinisikan ulang di beberapa cell | Satu fungsi global, dipanggil berkali-kali |
| **Kuantisasi Ternary** | Hanya FGQ (`ternary_quantize_group`) yang diimplementasikan | Dua varian: **Standard Ternary** (threshold tunggal global) + **FGQ** (threshold per grup) |
| **Eksperimen Group Size** | Perbandingan N = 2, 4, 8, 16, 32, 64 dengan loop eksplisit | Panggil fungsi dengan parameter `group_size` — satu baris per skenario |
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

**Skema Baru** — Panggil fungsi modular + baseline aman:
```python
model_fgq4, mean_alphas4 = apply_fgq_to_model(model_fp32, group_size=4)
acc_fgq4 = evaluate_accuracy(model_fgq4, test_loader, device)
```

---

## 🧪 Hasil Eksperimen (Contoh)

Setelah menjalankan notebook, hasil perbandingan akurasi akan tampil seperti berikut:

| Skema | Akurasi |
|---|---|
| Baseline FP32 | ~91-92% |
| Standard Ternary | ~X% |
| FGQ N=4 | ~X% |
| FGQ N=8 | ~X% |
| FGQ N=16 | ~X% |

> **Catatan:** Angka aktual bergantung pada riwayat training model di session tersebut.

---

## 🚀 Cara Menggunakan

```python
# 1. Setup environment
model_path, train_loader, test_loader = setup_fashion_mnist_environment()

# 2. Load / Train model baseline
model_fp32, history = load_and_prepare_model(LeNet5, model_path,
                                              train_loader, test_loader, device)

# 3. Standard Ternary Quantization
model_ternary, alphas = apply_standard_ternary_to_model(model_fp32)
acc_ternary = evaluate_accuracy(model_ternary, test_loader, device)

# 4. Fine-Grained Ternary Quantization (FGQ)
model_fgq, alphas = apply_fgq_to_model(model_fp32, group_size=4)
acc_fgq = evaluate_accuracy(model_fgq, test_loader, device)

# 5. Bandingkan
print(f"Baseline: {history['accuracy'][-1]:.2f}%")
print(f"Ternary:  {acc_ternary:.2f}%")
print(f"FGQ N=4:  {acc_fgq:.2f}%")
```

---

## 📂 Struktur Folder

```
Modular Pipeline/
├── Quantization_Fashion_MNIST.ipynb   # Notebook utama (modular)
├── readme.md                          # File ini
```

Notebook lain di folder induk (`tes-1-Net/`, `tes-2-LeNet5/`, `tes-3-ResNet18/`) merupakan eksperimen dengan skema lama (monolitik) untuk berbagai dataset dan arsitektur.
