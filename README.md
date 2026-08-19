# Plant Disease Detection — Final Project Pembelajaran Mesin 2

Eksperimen klasifikasi 38 kelas penyakit/kondisi sehat daun pada dataset PlantVillage menggunakan Custom CNN, MobileNetV3-Small, dan EfficientNet-B0.

- **Mahasiswa:** Farrel Ghozy Affifudin
- **NIM:** 452024611053
- **Kelas:** TI5 A2
- **Universitas:** Universitas Darussalam Gontor
- **Hardware eksperimen:** NVIDIA GeForce RTX 4060, mixed precision FP16

## File yang Dikumpulkan

`paper/452024611053-Farrel Ghozy Affifudin.pdf`

Paper menggunakan IEEE Conference A4 dua kolom dan berjumlah maksimal lima halaman sesuai `Instruksi Tugas.pdf`.

## Hasil Final

Dataset aktual berisi **54.305 citra**, 38 kelas, dengan stratified split:

- Training: 43.429
- Validation: 5.417
- Testing: 5.459

| Exp | Model | Strategi | LR | Best epoch | Test Acc | Precision macro | Recall macro | F1 macro |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E1 | Custom CNN (0,54M) | From scratch, baseline | 1e-3 | 14 | 95,09% | 94,85% | 93,17% | 93,39% |
| E2 | MobileNetV3-Small | ImageNet, frozen backbone | 1e-3 | 12 | 97,47% | 96,79% | 96,18% | 96,41% |
| E3 | EfficientNet-B0 | ImageNet, frozen backbone | 1e-3 | 11 | 94,98% | 94,17% | 93,19% | 93,48% |
| E4 | MobileNetV3-Small | ImageNet, full fine-tuning | 1e-4 | 6 | **98,99%** | **98,81%** | **98,15%** | **98,42%** |

E4 menjalankan 10 dari maksimum 12 epoch dan dihentikan oleh early stopping. E1–E3 mencapai batas epoch yang dikonfigurasi. `Best epoch` adalah epoch dengan validation loss terendah; checkpoint yang dievaluasi berasal dari epoch tersebut.

## Konfigurasi

- Framework: PyTorch/torchvision
- Optimizer: Adam
- Loss: `torch.nn.CrossEntropyLoss`
- Batch size: 32
- Early stopping: patience 4, monitor validation loss
- Scheduler: ReduceLROnPlateau, factor 0,5, patience 2, minimum LR 1e-6
- Input: 224×224
- Normalisasi: ImageNet mean/std
- Augmentasi training: random resized crop (0,8–1,0), horizontal flip, rotasi ±15°, dan color jitter
- Validation/testing: resize 256 lalu center crop 224
- Seed: 42

## Struktur Repository

```text
├── Instruksi Tugas.pdf
├── Final_Project_Pembelajaran_Mesin2.ipynb
├── paper/
│   ├── plant_disease_detection.tex
│   ├── plant_disease_detection.pdf
│   └── 452024611053-Farrel Ghozy Affifudin.pdf
└── v2/
    ├── scripts/
    │   ├── prepare_data.py
    │   ├── common.py
    │   └── run_experiment.py
    └── results/
        ├── training_report.ipynb
        ├── run_all.log
        └── e1/ ... e4/
```

Setiap folder `v2/results/e*/` berisi:

- `metrics.json` dan `test_metrics.json`
- `history.json`
- `training_curves.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `per_class_analysis.json`
- `top_errors.json`

`v2/results/run_all.log` adalah catatan kanonik yang dibentuk dari metrics/history final. Notebook utama dan `training_report.ipynb` membaca artifact final tersebut.

## Reproduksi

### Instalasi

Gunakan Python 3.10 atau 3.11. Buat virtual environment lalu instal dependency. Untuk GPU, pilih wheel PyTorch/CUDA yang sesuai dari [pytorch.org](https://pytorch.org/get-started/locally/) sebelum dependency lain.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Contoh CUDA 12.4; sesuaikan dengan environment:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Unduh dataset:

https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Siapkan split dan jalankan eksperimen:

```bash
python3 v2/scripts/prepare_data.py
python3 v2/scripts/run_experiment.py --exp e1 --epochs 15 --batch 32
python3 v2/scripts/run_experiment.py --exp e2 --epochs 12 --batch 32
python3 v2/scripts/run_experiment.py --exp e3 --epochs 12 --batch 32
python3 v2/scripts/run_experiment.py --exp e4 --epochs 12 --batch 32
```

Default path script:

- Dataset: `~/pm2_v2/data/plantvillage_split`
- Output: `~/pm2_v2/results`
- Checkpoint: `~/pm2_v2/models`

Path dapat diganti melalui argumen `--data-root`, `--out-root`, dan `--model-root`.

## Catatan Validitas

Seluruh angka pada paper final berasal dari artifact `metrics.json` dan checkpoint eksperimen nyata. Tidak ada metrik simulasi atau angka placeholder dalam hasil final.
