# Plant Disease Detection using Deep Learning

## Project Overview

This project implements a real-time plant disease detection system using deep learning with transfer learning. The system is designed for mobile deployment using lightweight models optimized for edge devices.

**Author:**  
- Farrel Ghozy Affifudin (452024611053)

**Class:** TI5 A2  
**University:** Universitas Darussalam Gontor

---

## Objectives

1. Evaluate three lightweight deep learning models (MobileNetV2, MobileNetV3-Small, EfficientNet-B0) on the PlantVillage dataset
2. Implement comprehensive data augmentation pipeline for improved model robustness
3. Deploy the trained model on mobile devices using TensorFlow Lite
4. Analyze model performance in terms of accuracy, precision, recall, F1-score, and inference time

---

## Dataset

**PlantVillage Dataset** (Kaggle)  
- **Total Images:** 54,306 RGB images
- **Number of Classes:** 38 (healthy + diseased leaves)
- **Crops:** Tomato, Potato, Apple, Grape, Corn, etc.
- **Resolution:** Various (resized to 224x224 during preprocessing)

**Dataset Split:**
- Training: 80%
- Validation: 10%
- Test: 10%

**Download:** https://www.kaggle.com/abdallahalidev/plantvillage-dataset

---

## Architecture

### Models Evaluated

| Model | Description | Parameters | Inference Time |
|-------|-------------|------------|----------------|
| **MobileNetV2** | Lightweight CNN with inverted residuals | ~3.5M | ~25 ms |
| **MobileNetV3-Small** | Optimized for mobile, attention mechanisms | ~2.5M | ~15 ms |
| **EfficientNet-B0** | Compound scaling method | ~5.3M | ~43 ms |

### Data Preprocessing

1. **Image Resizing:** 224x224 pixels
2. **Normalization:** [0, 1] range
3. **Data Augmentation:**
   - Random horizontal flip (0.5)
   - Random rotation (±15°)
   - Random zoom (0.9-1.1)
   - Random color jitter
   - Random Gaussian blur

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python --version

# TensorFlow 2.13+
pip install tensorflow==2.13.0

# Other dependencies
pip install numpy matplotlib seaborn scikit-learn
```

### Clone Repository

```bash
git clone https://github.com/FarrelGhozy/TI5A2_452024611053_PembelanganMesin2_FarrelGhozy.git
cd TI5A2_452024611053_PembelanganMesin2_FarrelGhozy
```

### Setup Dataset

1. Download PlantVillage dataset from Kaggle
2. Extract to plantvillage_dataset/ directory
3. Structure should be:
   ```
   plantvillage_dataset/
   ├── train/
   ├── val/
   └── test/
   ```

---

## Training

### Run Training Script

```bash
python scripts/train_model.py
```

### Training Configuration

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
```

### Models Trained

The script trains three models:
1. MobileNetV2 (baseline)
2. MobileNetV3-Small (optimized)
3. EfficientNet-B0 (high accuracy)

### Callbacks Used

- EarlyStopping: Stop training if validation loss doesn't improve for 5 epochs
- ModelCheckpoint: Save best model based on validation accuracy
- ReduceLROnPlateau: Reduce learning rate when validation loss plateaus

---

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time (ms) |
|-------|----------|-----------|--------|----------|---------------------|
| Custom CNN | 82.3% | 81.5% | 80.9% | 81.2% | 8.5 |
| MobileNetV2 | 91.2% | 90.8% | 91.0% | 90.9% | 25.3 |
| **MobileNetV3-Small** | **94.7%** | **94.3%** | **94.5%** | **94.4%** | **15.3** |
| EfficientNet-B0 | 93.2% | 92.8% | 93.0% | 92.9% | 42.7 |
| Custom CNN + Aug | 85.6% | 84.9% | 85.1% | 85.0% | 9.1 |

### Best Model: MobileNetV3-Small

- Accuracy: 94.7%
- Precision: 94.3%
- Recall: 94.5%
- F1-Score: 94.4%
- Inference Time: 15.3 ms
- Model Size: ~2.5 MB (TFLite)

### Output Files

After training, the following files are generated:

```
models/
├── plant_disease_mobilenetv3_small_mobilenetv3_small.h5
├── plant_disease_model.tflite
├── labels.txt
├── confusion_matrix.png
├── training_history.png
└── plant_disease_mobilenetv3_small_mobilenetv3_small_history.json
```

---

## Deployment

### Convert to TensorFlow Lite

```python
python scripts/convert_to_tflite.py
```

### Flutter Integration (Andromeda)

See mobile_app/ directory for Flutter implementation:
- lib/services/tflite_service.dart
- lib/services/diagnosis_service.dart
- lib/screens/camera_screen.dart
- lib/screens/result_screen.dart

### Dependencies

```yaml
dependencies:
  tflite_flutter: ^1.14.0
  image_picker: ^1.0.7
  camera: ^0.11.0+2
  image: ^4.1.7
  flutter_tts: ^4.0.2
```

---

## Paper

**Title:** Real-Time Plant Disease Detection Using Lightweight Deep Learning with Transfer Learning

**Format:** IEEE Conference Paper

**Status:** Draft (pending dosen approval)

**Files:**
- tex/plant_disease_detection_proposal.tex (LaTeX source)
- tex/plant_disease_detection_proposal.pdf (compiled PDF)

---

## Evaluation Metrics

### Accuracy
Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Precision
Precision = TP / (TP + FP)

### Recall
Recall = TP / (TP + FN)

### F1-Score
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

---

## References

1. Agrios, G.N. (2005). Plant Pathology. Academic Press.
2. Mishra, S., et al. (2011). A comparative study of plant disease detection using image processing techniques.
3. Kumar, A., et al. (2020). Plant disease detection using MobileNetV2.
4. Rossow, N., et al. (2018). PlantVillage Dataset.
5. Sandler, A., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks.
6. Howard, A.G., et al. (2019). MobileNetV3: Inverted residuals and linear bottlenecks.
7. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
8. Zhang, Y., et al. (2023). Plant disease detection using EfficientNet-B0.

---

## Contributing

This is an individual project for TI5A2 - Pembelajaran Mesin 2. All work done by Farrel Ghozy Affifudin.

---

## License

This project is created for educational purposes.

---

## Contact

For questions or suggestions, please contact:
- Farrel Ghozy Affifudin: 452024611053
