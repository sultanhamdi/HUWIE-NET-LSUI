# HuWie-Net-PINN — UIE Ablation Pipeline

Underwater image enhancement: **HuWie-Net backbone + Beer-Lambert PINN branch +
DINOv2 perceptual loss**. Ablation study 4 varian, train di Kaggle GPU,
output lengkap untuk laporan penelitian.

Backbone = official **HuWie-Net** (HUWIE-Net, ITU UIE-Lab). Three modules:
Image-to-Image (I2IM), Physics-Informed (PIM), dan Fusion. ~0.6M params.

| Variant | PINN | DINOv2 | Arti |
|---|---|---|---|
| A1 | — | — | HuWie-Net baseline |
| A2 | ✓ | — | + physics (Beer-Lambert) |
| A3 | — | ✓ | + perceptual (DINOv2) |
| A4 | ✓ | ✓ | full model |

Semua varian: epoch identik (50), batch identik (16), training set identik (LSUI 4279 pairs) → ablation adil.

**Optimisasi training:** AdamW + cosine LR schedule + warmup 3 epoch + EMA +
gradient clipping + bf16 mixed precision. I2IM+PIM pretrained di-freeze 2 epoch
awal (soft warmup) supaya PINN head yang random tidak merusak fitur pretrained,
lalu full fine-tune.

---

## Prasyarat

- Python ≥ 3.9
- Akun [Kaggle](https://www.kaggle.com) (untuk GPU gratis dan download dataset)
- **Kaggle API key** (`kaggle.json`) — lihat Langkah 0

---

## Langkah 0 — Setup satu kali (~10 menit)

### 0.1 Install dependencies

```bash
pip install torch torchvision einops pyyaml numpy pillow matplotlib \
    scikit-image scipy torchmetrics lpips tqdm kaggle
```

### 0.2 Dapatkan Kaggle API key

1. Buka [kaggle.com](https://www.kaggle.com) → klik foto profil → **Settings**
2. Scroll ke **API** → klik **Create New Token**
3. File `kaggle.json` otomatis ter-download. Letakkan di `~/.kaggle/kaggle.json`
   (Linux/Mac) atau `C:\Users\<user>\.kaggle\kaggle.json` (Windows).

### 0.3 Accept rules dataset Kaggle di browser

Buka ketiga link ini di browser, login, dan klik **"I Understand and Accept"**:

- [LSUI Dataset](https://www.kaggle.com/datasets/cbhavik/lsui-dataset)
- [UIEB Dataset Raw](https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw)
- [UIEB Dataset Reference](https://www.kaggle.com/datasets/larjeck/uieb-dataset-reference)

---

## Langkah 1 — Download dataset & pretrained (jalankan sekali)

```bash
python download.py
```

Script ini otomatis:
- Download LSUI (4279 pasang gambar) dari Kaggle
- Download UIEB raw + reference dari Kaggle
- Download pretrained HuWie-Net (~2.2 MB) dari GitHub
- Simpan semua ke folder `./data/`

**Output yang benar di akhir log:**
```
[download] counts: {'lsui_raw': 4279, 'lsui_ref': 4279, 'uieb_raw': 890, 'uieb_ref': 890}
[download] DONE
```

Re-run aman: kalau download sudah ada, script skip dan langsung `DONE`.

---

## Langkah 2 — Sanity check (opsional, ~3–5 menit)

Jalankan 1 epoch untuk memastikan GPU path, checkpoint, dan resume bekerja:

```bash
python train.py --variant A1 --smoke
```

Cek log-nya:
- Muncul `[pretrained] N/N checkpoint tensors loaded` → pretrained termuat
- Muncul `loss=...` dan `val_psnr=...` → GPU forward/backward bekerja
- Muncul `[data] train=4279 pairs  val=890 pairs` → dataset terbaca

Jalankan ulang untuk tes resume:
```bash
python train.py --variant A1 --smoke
```
Harus muncul `[resume] from epoch 1` → crash-safe resume bekerja.

---

## Langkah 3 — Training

### Opsi A: Jalankan satu per satu

```bash
python train.py --variant A1
python train.py --variant A2
python train.py --variant A3
python train.py --variant A4
```

### Opsi B: Jalankan semua sekaligus (A1 → A4 sekuensial otomatis)

```bash
python train.py --all
```

### Informasi penting

**Auto-resume:** Kalau run terputus, cukup jalankan perintah yang sama — training
lanjut dari checkpoint terakhir secara otomatis.

**Ganti num_epochs:** Edit `num_epochs` di `configs/base.yaml`. **Jangan override
per-varian** — harus identik di semua varian agar ablation adil.

---

## Langkah 4 — Evaluasi

```bash
python evaluate.py --all
```

Script ini:
- Load `best.pt` (EMA weights) tiap varian
- Hitung PSNR, SSIM, LPIPS, UCIQE, UIQM di seluruh UIEB-890 (dengan TTA 8-way)
- Tulis `metrics.json` tiap varian
- Buat tabel ablasi dan tabel perbandingan vs baseline

---

## Menjalankan di Kaggle Notebook

Cara termudah: upload kode ke GitHub, lalu di Kaggle Notebook:

```python
# Cell 1: Clone repo
!git clone https://github.com/sultanhamdi/HUWIE-NET-LSUI.git
%cd HUWIE-NET-LSUI

# Cell 2: Install deps
!pip install -q einops lpips torchmetrics

# Cell 3: Download dataset
!python download.py

# Cell 4: Training
!python train.py --variant A4

# Cell 5: Evaluate
!python evaluate.py --all
```

Atau tanpa GitHub: upload folder sebagai **Kaggle Dataset**, lalu unzip dan jalankan.

---

## Isi output (untuk laporan)

```
outputs/
├── ablation_summary.md          ← tabel ablasi (4 varian × semua metrik) → copy ke laporan
├── ablation_summary.csv         ← versi CSV untuk spreadsheet / LaTeX
├── comparison_table.md          ← ours vs baseline yang published
│
├── A1/
│   ├── config_snapshot.yaml     ← config persis yang dipakai (reproducibility)
│   ├── train_log.csv            ← per-epoch: loss, lr, val_psnr, val_ssim
│   ├── train_log.txt            ← full stdout training
│   ├── metrics.json             ← hasil evaluasi final di UIEB-890 (mean ± std)
│   ├── curves/
│   │   ├── loss.png             ← loss curve (total, charbonnier, physics, perceptual)
│   │   └── val_metrics.png      ← PSNR & SSIM per epoch
│   ├── samples/
│   │   └── grid.png             ← Input | Enhanced | Reference (8 gambar)
│   ├── physics/                 ← kosong untuk A1 (tidak ada PINN)
│   └── checkpoints/
│       ├── best.pt              ← model terbaik (EMA)
│       ├── latest.pt            ← checkpoint terakhir (untuk resume)
│       └── epoch_XXXX.pt        ← periodic checkpoint
│
├── A2/  (seperti A1 + physics/beta_depth.png dan beta_depth.csv)
├── A3/  (seperti A1)
└── A4/  (seperti A2)
```

### Mapping ke bagian laporan

| Output | Masuk ke bagian |
|---|---|
| `ablation_summary.md` | Tabel ablasi di bagian Experiment / Results |
| `comparison_table.md` | Tabel perbandingan vs prior work |
| `samples/grid.png` | Qualitative results figure |
| `curves/val_metrics.png` | Convergence plot di bagian Training |
| `curves/loss.png` | Loss curve di bagian Training |
| `physics/beta_depth.png` | Bukti PINN: β_R > β_G > β_B sesuai Beer-Lambert |
| `metrics.json` | Angka untuk isi tabel (PSNR/SSIM/LPIPS) |

---

## Troubleshooting

### Download gagal dengan 403 / HTTP Error

Kaggle rules belum di-accept. Buka ketiga link di Langkah 0.3 dan klik Accept.

### Pretrained tidak termuat

```
[pretrained] WARNING not found at ... — random init
```

Jalankan ulang `python download.py`. Kalau tetap gagal, download manual dari
[GitHub](https://github.com/UIE-Lab/HUWIE-Net/raw/main/pre_trained_models/HUWIE_Net_epoch50.pth)
lalu simpan ke `./data/pretrained/HUWIE_Net_epoch50.pth`.

### CUDA OOM

Turunkan `batch_size` di `configs/base.yaml` dan naikkan `accum_steps`:
| batch_size | accum_steps | eff. batch |
|---|---|---|
| 16 | 1 | 16 (default) |
| 8 | 2 | 16 |
| 4 | 4 | 16 |

### Training terputus di tengah jalan

Jalankan ulang perintah yang sama — training lanjut dari checkpoint terakhir.

---

## Struktur kode (referensi)

```
HUWIE-NET-LSUI/
├── README.md
├── download.py           ← python download.py
├── train.py              ← python train.py --variant A1|A2|A3|A4|--all
├── evaluate.py           ← python evaluate.py --all
├── configs/
│   ├── base.yaml         ← semua hyperparameter shared (edit di sini)
│   ├── A1.yaml           ← hanya: use_pinn: false, use_dino: false
│   ├── A2.yaml           ← hanya: use_pinn: true,  use_dino: false
│   ├── A3.yaml           ← hanya: use_pinn: false, use_dino: true
│   └── A4.yaml           ← hanya: use_pinn: true,  use_dino: true
└── src/
    ├── model.py          ← HuWieNetPINN (backbone + PINN branch)
    ├── pretrained.py     ← load checkpoint pretrained
    ├── data.py           ← UIEDataset + DataLoader
    ├── losses.py         ← Charbonnier + physics + DINOv2 perceptual
    ├── metrics.py        ← PSNR/SSIM/LPIPS/UCIQE/UIQM
    ├── trainer.py        ← training loop (EMA, AMP, resume, logging)
    ├── evaluator.py      ← eval loop + ablation/comparison tables
    ├── config.py         ← YAML loader
    └── assets.py         ← dataset folder detection
```
