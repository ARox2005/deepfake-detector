# DeepFake Image Detection

A deep learning project for detecting deepfake images using the **Xception** architecture, trained on the **Celeb-DF v2** dataset.

## Project Overview

This project uses transfer learning with a pretrained Xception model (via `timm`) to classify face images as **real** or **fake (AI-generated)**. The model is fine-tuned on frames extracted from the Celeb-DF v2 video dataset.

### Key Features
- Binary classification: Real vs Fake face images
- Xception backbone pretrained on ImageNet
- Class imbalance handling via weighted loss (`BCEWithLogitsLoss` with `pos_weight`)
- Top-K checkpoint saving by both validation accuracy and AUC
- Resumable training from any checkpoint
- Single image and video prediction support

## Dataset: Celeb-DF v2

This project uses the [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) dataset — a large-scale deepfake forensics benchmark.

### Dataset Composition
| Category | Videos | Description |
|---|---|---|
| Celeb-real | 590 | Real celebrity interview videos |
| Celeb-synthesis | 5,639 | DeepFake videos generated from real videos |
| YouTube-real | 300 | Real videos from YouTube |

### Data Preparation (not included in this repo)
Frames were extracted from the videos (5 per video) to create an image dataset for training. The frames were split by celebrity identity (not random) to prevent data leakage:

| Split | Images |
|---|---|
| Train | 22,881 (1,941 real + 20,940 fake) |
| Validation | 4,070 |
| Test | 4,140 |

The split metadata is stored as CSV files in `data/splitted_data/`:
- `images_train.csv`
- `images_val.csv`
- `images_test.csv`

Each CSV contains columns: `frame_id`, `video_id`, `target_id`, `source_id`, `label` (0=real, 1=fake), `frame_index`, `file_path`

## Project Structure

```
DeepFakeDetection/
├── config.py          # Constants, paths, transforms, hyperparameters
├── dataset.py         # DeepfakeDataset class (PyTorch Dataset)
├── model.py           # Model creation (Xception via timm)
├── train.py           # Training script with checkpoint management
├── test.py            # Testing/evaluation script
├── predict.py         # Single image & video prediction
├── data/
│   └── splitted_data/
│       ├── images_train.csv
│       ├── images_val.csv
│       └── images_test.csv
└── model/
    └── checkpoints/   # Saved checkpoints
```

## Requirements

- Python 3.10+
- PyTorch (with CUDA support recommended)
- timm
- scikit-learn
- tqdm
- pandas
- Pillow

Install dependencies:
```bash
pip install torch torchvision timm scikit-learn tqdm pandas Pillow
```

## Usage

### Training

```bash
# Train from scratch
python train.py

# Resume from latest checkpoint
python train.py --resume latest

# Resume from best AUC model
python train.py --resume best

# Resume from a specific checkpoint
python train.py --resume "model\checkpoints\checkpoint_5.pth"
```

Training saves:
- **Latest 3 epoch checkpoints** (`checkpoint_*.pth`) — older ones are automatically deleted
- **Top 2 by validation accuracy** (`top_k_epoch_*.pth`)
- **Top 2 by validation AUC** (`best_auc_epoch_*.pth`)

### Testing

```bash
python test.py
```

Loads the best AUC model and evaluates on the test set. Outputs test loss, accuracy, AUC-ROC, and confusion matrix.

### Prediction

```bash
# Predict on a single image
python predict.py "path\to\image.jpg"
```

## Model Architecture

- **Backbone**: Xception (pretrained on ImageNet)
- **Output**: Single logit → sigmoid → probability of being fake
- **Input size**: 299 × 299 RGB
- **Loss**: BCEWithLogitsLoss with pos_weight to handle class imbalance
- **Optimizer**: AdamW (lr=1e-4)

## Training Results

| Metric | Value |
|---|---|
| Best Val AUC | 0.9603 |
| Test AUC | 0.9574 |
| Test Accuracy | 87.46% |
