# Modular Pipeline — Quantization MNIST & Fashion MNIST

## 📌 Gambaran Umum

Folder ini berisi **dua notebook modular** yang menerapkan pendekatan modular untuk eksperimen kuantisasi model *ternary* — **Standard Ternary** dan **Fine-Grained Ternary Quantization (FGQ)**, masing-masing dengan varian **symmetric** maupun **asymmetric threshold** — pada dua dataset:

| Notebook | Dataset | Arsitektur | Section tambahan |
|---|---|---|---|
| `Quantization_MNIST.ipynb` | **MNIST** | LeNet-5 & ResNet-18 | — |
| `Quantization_Fashion_MNIST.ipynb` | **Fashion-MNIST** | LeNet-5, ResNet-18 & ResNet-18 Modified | Sel debug pack/unpack, ekspor TorchScript (`.pt`), inferensi C++ (LibTorch) |

Kedua notebook berbagi **blok fungsi modular yang identik** (setup dataset, training, kuantisasi, penyimpanan). Perbedaannya terletak pada fungsi setup dataset, kelas model, serta section tambahan di notebook Fashion-MNIST.

Pipeline ini merupakan hasil refaktorisasi dari skema monolitik sebelumnya — setiap tanggung jawab dipisahkan ke dalam fungsi independen yang dapat digunakan ulang, dan kedua notebook mengikuti **alur tahapan yang konsisten**.

---

## 🗂️ Alur Notebook (Struktur Section)

### `Quantization_MNIST.ipynb`

| # | Section | Isi |
|---|---|---|
| 1 | **LOAD & EDA MNIST** | Load & stratified split (50k train / 10k val / 10k test), EDA (sparsity, distribusi label, statistik dasar, visualisasi gambar, brightness & mean intensitas pixel) |
| 2 | **DEFINISI FUNGSI MODULAR** | Setup environment, trainer/evaluator, fungsi kuantisasi, definisi model, fungsi penyimpanan (safetensors), generate input batch |
| 3 | **PENGUJIAN TRAINING & KUANTISASI** | Model setup (ResNet-18 & LeNet-5), Standard Ternary (symmetric + asymmetric) per arsitektur, FGQ, perbandingan akurasi, penyimpanan & verifikasi safetensors |
| 4 | **INFERENSI** | Upload gambar, preprocessing, prediksi + benchmark FP32 vs kuantisasi (ResNet-18 & LeNet-5) |

### `Quantization_Fashion_MNIST.ipynb`

| # | Section | Isi |
|---|---|---|
| 1 | **LOAD & EDA FASHION MNIST** | Sama seperti notebook MNIST, untuk Fashion-MNIST |
| 2 | **DEFINISI FUNGSI MODULAR** | Sama, ditambah kelas **`ResNet18Modified`** |
| 3 | **PENGUJIAN TRAINING & KUANTISASI** | Model setup (ResNet-18 Modified, varian Modified, LeNet-5), Standard Ternary (symmetric + asymmetric) + **sel debug/verifikasi**, FGQ dengan group size lebih banyak, perbandingan akurasi, penyimpanan safetensors, **verifikasi load**, **konversi ke TorchScript (`.pt`)** |
| 4 | **INFERENSI** | Upload gambar + benchmark untuk ResNet-18 Standard, ResNet-18 Fine-Grained, LeNet-5 Standard, LeNet-5 Fine-Grained |
| 5 | **INFERENSI DENGAN CPP** | Setup LibTorch 2.4.0, build proyek C++ via `cmake`, jalankan inferensi `.pt` dengan binary `./build/inference` |

---

## 🧩 Blok Fungsi Modular

### A. Setup & Training

| Fungsi | Signature | Tanggung Jawab |
|---|---|---|
| `setup_mnist_environment()` | `(model_name, force_remount=True)` | Mount Google Drive, siapkan path model, buat `train_loader` & `test_loader` **MNIST** |
| `setup_fashion_mnist_environment()` | `(model_name, force_remount=True)` | Idem untuk **Fashion-MNIST** |
| `evaluate_accuracy()` | `(model, data_loader, device)` | Evaluasi akurasi top-1 pada data loader |
| `load_and_prepare_model()` | `(model_class, model_path, train_loader, test_loader, device, num_epochs=10, lr=0.001)` | Load model dari checkpoint (jika ada) **atau** training dari awal dengan **best-model tracking**, lalu simpan `state_dict` + `history` |
| `get_best_accuracy()` | `(history, metric='accuracy')` | Ringkas best/final accuracy, best epoch, dan best loss dari `history` |

### B. Kuantisasi

| Fungsi | Signature | Tanggung Jawab |
|---|---|---|
| `standard_ternary_quantize()` | `(weight_tensor, asymmetric=False)` | Kuantisasi ternary standar per tensor (threshold `Δ = 0.7 × mean(\|W\|)`), symmetric atau asymmetric |
| `apply_standard_ternary_to_model()` | `(model, asymmetric=False, verbose=False, save_path=None, skip_conv1=False)` | Terapkan ternary standar ke seluruh layer Conv2d/Linear via `copy.deepcopy()`; opsi `skip_conv1` & `save_path` |
| `ternary_quantize_group()` | `(weight_tensor, group_size=4, asymmetric=False)` | Kuantisasi per grup (FGQ) — mode **symmetric** (optimasi 20 kandidat threshold) **atau asymmetric** (threshold & alpha terpisah untuk nilai positif/negatif) |
| `apply_fgq_to_model()` | `(model, group_size=4, asymmetric=False, verbose=False, save_path=None)` | Terapkan FGQ ke seluruh layer; `group_size` boleh satu angka **atau himpunan** (mis. `{4, 8, 16}`) untuk batch eksperimen |
| `find_best_fgq_model()` | `(fgq_results, test_loader, device, save_log_path=None)` | Evaluasi semua group size, kembalikan yang akurasinya terbaik + log akurasi (JSON) |

### C. Penyimpanan & Pemuatan (Safetensors)

| Fungsi | Signature | Tanggung Jawab |
|---|---|---|
| `pack_2bit()` / `unpack_2bit()` | `(mask)` / `(packed, shape, numel)` | Pack/unpack 4 nilai ternary (0/1/2) ke dalam 1 byte |
| `get_alpha_from_weight()` | `(weight_tensor)` | Ekstrak satu alpha global — dipertahankan untuk debugging/kompatibilitas |
| `get_group_alphas_from_weight()` | `(w_flat, group_size)` | Ekstrak **alpha per grup** (positif & negatif terpisah) langsung dari bobot ter-kuantisasi; aman untuk symmetric, asymmetric, maupun standard ternary (`group_size = numel`) |
| `save_ternary_safetensors()` | `(model, alphas=None, group_size=None, path=..., skip_layers=None)` | Simpan mask ternary (packed) + alpha per grup + BatchNorm + bias ke `.safetensors`; `skip_layers` untuk mempertahankan layer FP32 (mis. `model.conv1`) |
| `load_ternary_safetensors()` | `(path, template_model, verbose=True)` | Rekonstruksi model dari `.safetensors`, broadcast alpha per grup, verifikasi `max_diff`, dan tetap mendukung format lama (single-alpha) |

### D. Definisi Model

| Kelas | Notebook | Keterangan |
|---|---|---|
| `LeNet5` | Keduanya | 2 layer Conv + 3 layer FC |
| `ResNet18` | Keduanya | ResNet-18 dengan `conv1` diubah ke 1 channel |
| `ResNet18Modified` | Fashion-MNIST | Varian `conv1` 3×3 stride 1 + `maxpool` diganti `Identity`, cocok untuk citra 28×28 |

---

## ⚖️ Standard Ternary vs Fine-Grained Ternary Quantization

### Standard Ternary

Satu threshold (`Δ = 0.7 × mean(|W|)`) dan satu alpha untuk **seluruh layer** (per tensor). Mode symmetric maupun asymmetric didukung melalui parameter `asymmetric`.

### FGQ

Kuantisasi dilakukan **per grup** sepanjang `group_size` elemen:

- **Symmetric** — satu `delta` dan satu `alpha` per grup (threshold dioptimasi dari 20 kandidat).
- **Asymmetric** — threshold & alpha positif/negatif dihitung terpisah (`d_pos`/`a_pos` vs `d_neg`/`a_neg`):
  - `d_pos = 0.7 × mean(pos_vals)`, lalu `a_pos = mean(pos_vals[pos_vals > d_pos])`
  - `d_neg = 0.7 × mean(|neg_vals|)`, lalu `a_neg = mean(neg_vals[neg_vals < -d_neg])`
  - Bobot yang tidak melewati threshold masing-masing di-nol-kan.

```python
# Standard Ternary (symmetric)
model_ternary, alphas = apply_standard_ternary_to_model(model_fp32, asymmetric=False)

# Standard Ternary (asymmetric)
model_ternary_asym, _ = apply_standard_ternary_to_model(model_fp32, asymmetric=True)

# FGQ symmetric
model_fgq_sym, _ = apply_fgq_to_model(model_fp32, group_size=4, asymmetric=False)

# FGQ asymmetric — beberapa group size sekaligus
fgq_results = apply_fgq_to_model(model_fp32, group_size={4, 8, 16}, asymmetric=True)
best_gs, best_model, best_acc = find_best_fgq_model(fgq_results, test_loader, device)
```

Dalam eksperimen, **FGQ dijalankan dengan `asymmetric=True`** pada kedua notebook.

---

## 💾 Penyimpanan Model & Verifikasi (Safetensors)

Setiap model hasil kuantisasi disimpan sebagai `.safetensors` dengan skema:

- **Mask ternary** di-*pack* 4 nilai (0/1/2) ke dalam 1 byte (`pack_2bit`).
- **Alpha per grup** disimpan sebagai dua tensor terpisah (`alpha_pos`, `alpha_neg`, dtype `float16`) — sehingga symmetric maupun asymmetric tetap akurat saat di-load.
- **BatchNorm** (weight/bias/running_mean/running_var) dan **bias** layer Conv/Linear tetap disimpan penuh.
- Layer yang dikecualikan (mis. `model.conv1`) dapat dipertahankan FP32 via `skip_layers`.

Setelah penyimpanan, notebook menjalankan **verifikasi round-trip**: akurasi sebelum `save` dibandingkan dengan akurasi setelah `load` (harus sama), dan metadata layer ditampilkan.

Notebook Fashion-MNIST menambahkan sel debug: `debug_layer_pack_unpack`, `test_pack_unpack_random`, `verify_ternary_weights`, `verify_mask_values`, `debug_alpha_comparison`, `debug_model_layers`, serta penyimpanan tanpa packing sebagai pembanding.

---

## 🆚 Perbedaan Kedua Notebook

| Aspek | `Quantization_MNIST.ipynb` | `Quantization_Fashion_MNIST.ipynb` |
|---|---|---|
| **Dataset** | MNIST | Fashion-MNIST |
| **Arsitektur** | LeNet-5, ResNet-18 | LeNet-5, ResNet-18, **ResNet-18 Modified** |
| **Group size FGQ** | `{4, 8, 16}` | ResNet-18 `{4, 8, 16, 32}`, LeNet-5 `{2, 3, 4, 8, 16, 32, 64}` |
| **Sel debug pack/unpack** | — | Ada (verifikasi reversibilitas packing, cek alpha, cek mask, simpan tanpa packing) |
| **Ekspor TorchScript (`.pt`)** | — | Ada (`torch.jit.trace` / `torch.jit.script`) |
| **Inferensi C++ (LibTorch)** | — | Ada (download LibTorch 2.4.0, build `cmake`, jalankan `./build/inference`) |
| **Normalisasi input** | `0.1307 / 0.3081` (standar MNIST) | `0.2860 / 0.3530` (standar Fashion-MNIST); sebagian eksperimen LeNet-5 memakai `0.5 / 0.5` |

### Perbedaan dengan Skema Lama (Monolitik)

| Aspek | Skema Lama | Skema Modular Sekarang |
|---|---|---|
| **Struktur Notebook** | Kode inline dan berulang | Fungsi modular di bagian definisi, eksekusi terpisah di akhir |
| **Dataset** | *Custom dataset* dari Drive (`HandwrittenDigitDataset`) | Dataset standar **torchvision** (MNIST & Fashion-MNIST) + **stratified split** |
| **Model** | Didefinisikan berkali-kali | Class `LeNet5` / `ResNet18` / `ResNet18Modified` terpusat |
| **Manajemen Checkpoint** | `state_dict` saja | `state_dict` + `history` (loss & accuracy per epoch) + **best-model tracking** |
| **Keamanan Model Asli** | Kuantisasi memodifikasi model langsung | `copy.deepcopy()` — model FP32 tetap aman |
| **Evaluasi** | Didefinisikan ulang di banyak cell | Satu fungsi `evaluate_accuracy()` global |
| **Kuantisasi** | Hanya FGQ | **Standard Ternary** + **FGQ**, masing-masing symmetric & asymmetric |
| **Penyimpanan** | Belum terstruktur | `.safetensors` ter-*pack* 2-bit + alpha per grup + verifikasi round-trip |
| **Reusability** | Rendah | Tinggi — fungsi dipakai ulang lintas dataset & arsitektur |

### Contoh Kode

**Standard Ternary + verifikasi round-trip:**
```python
model_ternary, alphas = apply_standard_ternary_to_model(
    model_fp32, asymmetric=False, save_path=".../resnet18_standard.safetensors")
acc = evaluate_accuracy(model_ternary, test_loader, device)

model_loaded = load_ternary_safetensors(".../resnet18_standard.safetensors", ResNet18()).to(device)
print(evaluate_accuracy(model_loaded, test_loader, device))  # harus sama dengan `acc`
```

**FGQ untuk beberapa group size sekaligus:**
```python
fgq_results = apply_fgq_to_model(model_fp32, group_size={4, 8, 16}, asymmetric=True)
best_gs, best_model, best_acc = find_best_fgq_model(fgq_results, test_loader, device)
```

---

## 🧪 Hasil Eksperimen (Contoh)

Setelah menjalankan notebook, hasil perbandingan akurasi tampil untuk **tiap arsitektur** seperti berikut:

| Skema | LeNet-5 | ResNet-18 |
|---|---|---|
| Baseline FP32 | ~99% (MNIST) / ~91% (Fashion-MNIST) | ~99% (MNIST) / ~91% (Fashion-MNIST) |
| Standard Ternary (symmetric) | ~X% | ~X% |
| Standard Ternary (asymmetric) | ~X% | ~X% |
| FGQ (asymmetric, group size terbaik) | ~X% | ~X% |

> **Catatan:** Angka aktual bergantung pada riwayat training model di session tersebut. FGQ dijalankan dengan `asymmetric=True`.

---

## 🚀 Cara Menggunakan

```python
# 1. Setup environment (pilih sesuai notebook / dataset)
model_path, train_loader, test_loader = setup_mnist_environment('lenet5_mnist_baseline.pth')
# atau: model_path, ... = setup_fashion_mnist_environment('lenet5_fashion_baseline.pth')

# 2. Load / Train model baseline (dengan best-model tracking)
model_fp32, history = load_and_prepare_model(LeNet5, model_path,
                                              train_loader, test_loader, device,
                                              num_epochs=5)
best_info = get_best_accuracy(history)

# 3. Standard Ternary Quantization (+ simpan)
model_ternary, alphas = apply_standard_ternary_to_model(
    model_fp32, asymmetric=False, save_path=".../lenet5_standard.safetensors")
acc_ternary = evaluate_accuracy(model_ternary, test_loader, device)

# 4. Fine-Grained Ternary Quantization (FGQ) — asymmetric, beberapa group size
fgq_results = apply_fgq_to_model(model_fp32, group_size={4, 8, 16}, asymmetric=True)
best_gs, best_model, best_acc = find_best_fgq_model(fgq_results, test_loader, device)

# 5. Simpan & verifikasi round-trip
save_ternary_safetensors(best_model, group_size=best_gs, path=".../lenet5_fgq.safetensors")
model_loaded = load_ternary_safetensors(".../lenet5_fgq.safetensors", LeNet5()).to(device)

# 6. Bandingkan
print(f"Baseline: {best_info['best_accuracy']:.2f}%")
print(f"Ternary:  {acc_ternary:.2f}%")
print(f"FGQ (GS={best_gs}): {best_acc:.2f}%")
```

> **Catatan:** Seluruh path (`/content/drive/MyDrive/model_modular/...`) dan `google.colab` mengasumsikan eksekusi di **Google Colab** dengan Google Drive ter-mount.

---

## 📂 Struktur Folder

```
Modular Pipeline/
├── Quantization_Fashion_MNIST.ipynb   # Notebook modular — Fashion-MNIST (+ debug, TorchScript, C++)
├── Quantization_MNIST.ipynb           # Notebook modular — MNIST
└── readme.md                          # File ini
```

Notebook lain di folder induk (`tes-1-Net/`, `tes-2-LeNet5/`, `tes-3-ResNet18/`) merupakan eksperimen dengan skema lama (monolitik) untuk berbagai dataset dan arsitektur.
