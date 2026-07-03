# 🛡️ Explainable DeepFake Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

**An AI-powered DeepFake detection system using XceptionNet with Grad-CAM explainability,
frame-wise video analysis, and a modern Streamlit dashboard.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Datasets](#-datasets) • [Models](#-models) • [Dashboard](#-dashboard)

</div>

---

## 🎯 Project Overview

This project is a **production-grade Final Year Internship Project** that detects whether an uploaded image or video is a DeepFake. Unlike traditional classifiers, this system provides:

- **Explainable AI** via Grad-CAM heatmaps
- **Frame-wise video analysis** with temporal smoothing
- **Real-time webcam detection**
- **Batch processing** for multiple files
- **PDF report generation**
- **Detection history** via SQLite
- **Interactive analytics** with Plotly

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Detection** | Upload any image → face crop → model inference → Grad-CAM |
| 🎬 **Video Detection** | Frame-by-frame analysis with temporal smoothing & majority voting |
| 📡 **Webcam Detection** | Live frame capture and real-time prediction |
| 📦 **Batch Detection** | Process multiple images/videos at once with CSV export |
| 🔬 **Grad-CAM** | Visual explanation of which facial regions triggered detection |
| 📊 **Analytics** | Confidence timelines, frame distributions, model comparison |
| 📄 **PDF Reports** | Professional reports with Grad-CAM images and statistics |
| 🗄️ **History** | Full detection log stored in SQLite with search and filter |
| 🤖 **3 Models** | XceptionNet, EfficientNet-B0, ResNet50 — best auto-selected |
| ⚡ **CPU Support** | Runs fully on CPU; GPU acceleration used when available |

---

## 🏗️ Architecture

```
Input (Image / Video / Webcam)
        │
        ▼
┌─────────────────┐
│  Face Detection │  MediaPipe (primary) → Haar Cascade (fallback)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  Resize 299×299 · Normalize · Augmentation
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│           XceptionNet               │  Pre-trained on ImageNet
│  ┌──────────────────────────────┐   │  Fine-tuned on:
│  │  Depthwise Separable Convs   │   │  · FaceForensics++
│  │  GlobalAveragePooling        │   │  · Celeb-DF v2
│  │  Dense(512) + Dropout(0.5)   │   │  · DFDC
│  │  Dense(1) → Sigmoid          │   │
│  └──────────────────────────────┘   │
└────────┬────────────────────────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌────────────────┐   ┌─────────────────────┐
│   Prediction   │   │      Grad-CAM        │
│  FAKE / REAL   │   │  Heatmap + Overlay   │
│  + Confidence  │   │  Gradient Strength   │
└────────┬───────┘   └──────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │  Streamlit Dashboard│
         │  · PDF Report       │
         │  · SQLite History   │
         │  · Plotly Charts    │
         └─────────────────────┘
```

---

## 📁 Project Structure

```
deepfake_detection/
│
├── assets/                    # Static assets (logo, icons)
├── configs/
│   ├── config.yaml            # Central configuration
│   └── logging_config.yaml    # Logging setup
├── data/
│   ├── raw/                   # Downloaded raw datasets
│   ├── processed/             # Face-cropped images
│   └── metadata/              # CSV splits (train/val/test)
├── preprocessing/
│   ├── dataset_validator.py   # File integrity + duplicate detection
│   ├── face_extractor.py      # MediaPipe + Haar face detection
│   ├── augmentor.py           # Albumentations augmentation
│   └── pipeline.py            # Full preprocessing orchestrator
├── models/
│   ├── xceptionnet.py         # XceptionNet builder
│   ├── efficientnet.py        # EfficientNet-B0 builder
│   ├── resnet50.py            # ResNet50 builder
│   └── model_factory.py      # Factory + save/load
├── training/
│   ├── data_loader.py         # tf.data pipeline
│   ├── trainer.py             # Training loop + callbacks
│   └── progressive_trainer.py # 3-phase progressive training
├── evaluation/
│   ├── metrics.py             # All evaluation metrics
│   └── evaluator.py           # Multi-model evaluation
├── inference/
│   ├── image_detector.py      # Single image prediction
│   └── video_detector.py      # Frame-wise video prediction
├── gradcam/
│   └── grad_cam.py            # Grad-CAM implementation
├── webcam/
│   └── webcam_detector.py     # Real-time webcam detection
├── dashboard/
│   ├── pages/                 # 8 Streamlit pages
│   ├── components/            # Reusable UI components
│   └── static/style.css      # Dark mode CSS
├── reports/
│   └── pdf_generator.py       # ReportLab PDF generation
├── database/
│   ├── schema.py              # SQLite schema
│   └── repository.py          # CRUD operations
├── utils/
│   ├── logger.py              # Centralized logging
│   ├── file_utils.py          # File helpers
│   ├── image_utils.py         # Image processing
│   └── video_utils.py         # Video processing
├── tests/                     # Unit tests (pytest)
├── outputs/                   # Model weights, reports, charts
├── logs/                      # Application logs
├── app.py                     # Streamlit entrypoint
├── train.py                   # Training CLI
├── requirements.txt           # Pinned dependencies
└── .gitignore
```

---

## 🚀 Installation

### Prerequisites

- Python **3.11.x** (other versions are not supported)
- Git
- ~4 GB free disk space (for model weights + datasets)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection/deepfake_detection
```

### Step 2 — Create Virtual Environment

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify Installation

```bash
py -3.11 -m pytest tests/test_database.py -v
```

---

## 📊 Datasets

Download and place datasets in the `data/raw/` directory:

| Dataset | Size | Path | Link |
|---------|------|------|------|
| FaceForensics++ | ~100 GB | `data/raw/ff_plus_plus/` | [Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23) |
| Celeb-DF v2 | ~3.5 GB | `data/raw/celeb_df/` | [Kaggle](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) |
| DFDC | ~470 GB | `data/raw/dfdc/` | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |

Each dataset directory must follow this structure:
```
data/raw/<dataset_name>/
    real/    ← real images or videos
    fake/    ← fake/manipulated images or videos
```

---

## 🏋️ Training

### Preprocess datasets

```bash
py -3.11 train.py --preprocess --datasets celeb_df
# or for all datasets:
py -3.11 train.py --preprocess
```

### Train with progressive strategy

```bash
# Train XceptionNet (all 3 phases)
py -3.11 train.py --model xceptionnet --phases all

# Train only Phase 1
py -3.11 train.py --model xceptionnet --phases phase_1

# Train all models
py -3.11 train.py --model all
```

### Evaluate models

```bash
py -3.11 train.py --evaluate
```

---

## 🌐 Dashboard

```bash
# From the deepfake_detection/ directory:
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> ⚠️ **Demo Mode**: If no trained weights are found in `outputs/weights/`, the dashboard
> launches with randomly-initialised ImageNet weights. All UI features are fully functional
> — only predictions will be random until real weights are trained.

### Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | KPI cards, recent detections, system architecture |
| 🖼️ Image Detection | Upload image → Grad-CAM → PDF report |
| 🎬 Video Detection | Frame-wise analysis → charts → PDF report |
| 📡 Webcam | Live frame capture and detection |
| 📦 Batch | Multi-file upload, CSV export |
| 📄 Reports | Browse and download all PDF reports |
| 🕒 History | Searchable detection log with analytics |
| ℹ️ About | Tech stack, datasets, model performance |

---

## 🤖 Models

| Model | Input Size | Params | Notes |
|-------|-----------|--------|-------|
| **XceptionNet** ⭐ | 299×299 | ~22M | Primary model, best AUC |
| EfficientNet-B0 | 299×299 | ~5M | Fastest inference |
| ResNet50 | 299×299 | ~25M | Strong baseline |

### Training Strategy (Progressive)

```
Phase 1: FaceForensics++ only          → Freeze base, train head
Phase 2: FF++ + Celeb-DF               → Unfreeze last 20 layers
Phase 3: FF++ + Celeb-DF + DFDC        → Full fine-tune
```

---

## 📈 Evaluation Metrics

The system computes the following on the held-out test split:

- **Accuracy** — Overall correct predictions
- **Precision** — Of all FAKE predictions, how many were correct
- **Recall** — Of all actual FAKEs, how many were caught
- **F1 Score** — Harmonic mean of Precision and Recall
- **ROC AUC** — Area under the ROC curve (primary ranking metric)
- **Confusion Matrix**
- **Classification Report**

---

## 🧪 Running Tests

```bash
# All tests
py -3.11 -m pytest tests/ -v

# Specific test file
py -3.11 -m pytest tests/test_database.py -v
py -3.11 -m pytest tests/test_preprocessing.py -v

# With coverage
py -3.11 -m pytest tests/ --cov=. --cov-report=html
```

---

## 🔧 Configuration

All settings are centralised in [`configs/config.yaml`](configs/config.yaml):

```yaml
preprocessing:
  image_size: [299, 299]
  face_margin: 0.3
  split_ratios: {train: 0.70, val: 0.15, test: 0.15}

training:
  batch_size: 16
  initial_learning_rate: 0.0001
  max_epochs: 50

inference:
  confidence_threshold: 0.5
  video_frame_sample_rate: 5
  max_video_frames: 200
```

---

## 📄 PDF Reports

Each detection can generate a professional PDF report containing:

- File name, prediction, confidence
- Processing time and timestamp
- Model used
- Grad-CAM overlay image
- Frame-wise statistics (videos)
- Confidence bar chart

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.15 + Keras |
| Primary Model | XceptionNet |
| Face Detection | MediaPipe + OpenCV |
| Explainability | Grad-CAM |
| Dashboard | Streamlit 1.35 |
| Charts | Plotly + Matplotlib |
| Data | NumPy + Pandas |
| ML Utilities | Scikit-learn |
| Database | SQLite |
| PDF | ReportLab |
| Augmentation | Albumentations |
| Testing | pytest |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al., 2019
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) — Li et al., 2020
- [DFDC](https://ai.meta.com/datasets/dfdc/) — Dolhansky et al., 2020
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2020
- [XceptionNet](https://arxiv.org/abs/1610.02357) — Chollet, 2017

---

<div align="center">
Built with ❤️ as a Final Year Internship Project
</div>
