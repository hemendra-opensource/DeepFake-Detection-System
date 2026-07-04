<div align="center">

<img src="assets/diagrams/workflow_overview.png" alt="DeepFake Detection System Workflow" width="100%"/>

<br/>

# 🧠 Explainable DeepFake Detection
### Using XceptionNet · Grad-CAM · Frame-wise Video Analysis

*A production-grade AI system for detecting manipulated media with visual explainability*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.15-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00A886?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![XceptionNet](https://img.shields.io/badge/Model-XceptionNet-8A2BE2?style=for-the-badge&logo=neural-network&logoColor=white)]()
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-Transfer_Learning-0099CC?style=for-the-badge)]()
[![Grad-CAM](https://img.shields.io/badge/XAI-Grad--CAM-success?style=for-the-badge)]()

<br/>

[🚀 Quick Start](#-installation-guide) · [📖 Documentation](#-project-overview) · [📊 Features](#-features) · [🏗️ Architecture](#️-system-workflow) · [📸 Dashboard Preview](#-dashboard-preview)

---

</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [System Workflow](#️-system-workflow)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Datasets](#-datasets)
- [Model Details](#-model-details)
- [Dashboard Preview](#-dashboard-preview)
- [Performance Metrics](#-performance-metrics)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Project Overview

<details open>
<summary><b>Click to expand</b></summary>

### What is a DeepFake?

**DeepFake** refers to synthetic media — images, videos, or audio — in which a person's likeness is algorithmically swapped, altered, or fabricated using deep learning techniques such as **Generative Adversarial Networks (GANs)**, **autoencoders**, and **diffusion models**. The term is a portmanteau of "deep learning" and "fake."

Modern DeepFakes can produce hyper-realistic manipulations indistinguishable to the naked eye — posing serious threats to:
- 🔐 **Identity theft and fraud**
- 🗳️ **Political misinformation and propaganda**
- 📰 **Fake news and media manipulation**
- ⚖️ **Legal evidence tampering**
- 🧒 **Non-consensual synthetic media (NCSM)**

### Why is DeepFake Detection Critical?

As generative AI advances exponentially, the volume and sophistication of synthetic media has outpaced human ability to detect it. Automated detection systems powered by deep learning are now essential infrastructure for:

- Social media platforms enforcing content integrity
- Governments and law enforcement verifying media authenticity
- Journalists and fact-checkers validating sources
- Organizations protecting their executives from impersonation

### Why Explainable AI (Grad-CAM)?

Black-box AI models, while powerful, produce predictions without justification — critically limiting trust in high-stakes scenarios. This project integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate visual heatmaps that highlight *exactly which facial regions* triggered the model's decision.

This transforms the system from a binary classifier into a **transparent, auditable, and explainable AI tool** that satisfies both technical and non-technical stakeholders.

### Project Objectives

| # | Objective |
|---|-----------|
| 1 | Build a production-grade DeepFake detection system using Transfer Learning on XceptionNet |
| 2 | Implement Grad-CAM visual explainability for model predictions |
| 3 | Support image, video (frame-wise), webcam, and batch detection pipelines |
| 4 | Provide a modern, interactive AI dashboard built with Streamlit |
| 5 | Generate automated PDF/CSV detection reports with evidence |
| 6 | Maintain complete detection history with SQLite persistence |
| 7 | Ensure mathematically consistent confidence scores across all detection modes |

### Real-World Applications

- 🏦 **KYC Verification** — Banks and fintech companies verifying customer identity
- 🎬 **Media Integrity** — News agencies authenticating video content
- 🔏 **Digital Forensics** — Legal investigation of manipulated evidence
- 📱 **Social Platforms** — Automated content moderation at scale
- 🎓 **Academic Research** — Benchmark evaluation of detection methods

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## ✨ Features

<details open>
<summary><b>Click to expand</b></summary>

| # | Feature | Description |
|---|---------|-------------|
| ✅ | **Image DeepFake Detection** | Upload any image; face is automatically extracted, preprocessed, and classified |
| ✅ | **Video DeepFake Detection** | Full video analysis with frame-by-frame inference and temporal aggregation |
| ✅ | **Frame-wise Video Analysis** | Per-frame confidence timeline with visual frame gallery |
| ✅ | **Explainable AI — Grad-CAM** | Heatmap overlay highlighting manipulated facial regions |
| ✅ | **Confidence Score** | Mathematically consistent real/fake probability with confidence percentage |
| ✅ | **Confidence Visualization** | Interactive gauge chart and probability bar charts via Plotly |
| ✅ | **Webcam Detection** | Live real-time detection with auto-discovery of available camera indices |
| ✅ | **Batch Detection** | Multi-file queue processing with progress tracking and summary export |
| ✅ | **Detection History** | SQLite-backed history with search, filter, and export capabilities |
| ✅ | **PDF Report Generation** | Automated professional PDF reports with Grad-CAM and metadata |
| ✅ | **CSV Export** | Export batch or history results as structured CSV |
| ✅ | **Modern Streamlit Dashboard** | SaaS-grade UI with glassmorphism, dark/light theme, and micro-animations |
| ✅ | **Analytics Dashboard** | Session statistics, detection trends, and KPI cards |
| ✅ | **Settings & Configuration** | Threshold, confidence, and UI preferences |
| ✅ | **Responsive UI** | Adaptive layout for different screen sizes |

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 🏗️ System Workflow

<details open>
<summary><b>End-to-End Detection Pipeline</b></summary>

<br/>

<div align="center">
<img src="assets/diagrams/workflow_overview.png" alt="End-to-End System Workflow" width="100%"/>
<br/><em>Figure 1 — End-to-End DeepFake Detection System Workflow</em>
</div>

<br/>

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                   │
│            Image · Video · Webcam · Batch Upload                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                   INPUT VALIDATION                               │
│         Format Check · File Size · Type Verification            │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                   FACE DETECTION                                 │
│           MediaPipe · Bounding Box · Alignment                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                 IMAGE PREPROCESSING                              │
│     Crop · Resize 224×224 · Normalize · BGR→RGB               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│               XCEPTIONNET INFERENCE                              │
│    Pretrained Weights · Feature Extraction · Softmax Output     │
└─────────────┬────────────────────────────┬───────────────────────┘
              ↓                            ↓
┌─────────────────────────┐   ┌────────────────────────────────────┐
│     PREDICTION ENGINE   │   │        GRAD-CAM ENGINE             │
│  Real / Fake · Prob · CI│   │  Gradient Map · Heatmap · Overlay  │
└─────────────┬───────────┘   └────────────────┬───────────────────┘
              └──────────────┬─────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                  ANALYTICS & RESULTS                             │
│       Dashboard · KPI Cards · Charts · Frame Timeline           │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│               PERSISTENCE & REPORTING                            │
│     SQLite History · PDF Report · CSV Export · Download         │
└──────────────────────────────────────────────────────────────────┘
```

</details>

<details>
<summary><b>Training Pipeline</b></summary>

<br/>

<div align="center">
<img src="assets/diagrams/training_pipeline.png" alt="Training Pipeline" width="100%"/>
<br/><em>Figure 2 — Deep Learning Training Pipeline: DeepFake Detection with XceptionNet</em>
</div>

</details>

<details>
<summary><b>Batch Detection Workflow</b></summary>

<br/>

<div align="center">
<img src="assets/diagrams/batch_workflow.png" alt="Batch Detection Workflow" width="100%"/>
<br/><em>Figure 3 — Batch DeepFake Detection Workflow</em>
</div>

</details>

<details>
<summary><b>Report Generation Workflow</b></summary>

<br/>

<div align="center">
<img src="assets/diagrams/report_workflow.png" alt="Report Generation Workflow" width="100%"/>
<br/><em>Figure 4 — Automatic Report Generation Workflow</em>
</div>

</details>

<details>
<summary><b>Performance Evaluation Pipeline</b></summary>

<br/>

<div align="center">
<img src="assets/diagrams/metrics_pipeline.png" alt="Performance Metrics Pipeline" width="100%"/>
<br/><em>Figure 5 — Model Evaluation and Performance Metrics Pipeline</em>
</div>

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 🛠️ Tech Stack

<details open>
<summary><b>Click to expand</b></summary>

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.11.x | Core development language |
| **Frontend** | Streamlit | 1.35.0 | Interactive web dashboard |
| **Deep Learning** | TensorFlow | 2.15.1 | Model training & inference |
| **Neural Network API** | Keras | 2.15.0 | High-level model construction |
| **Computer Vision** | OpenCV | 4.9.0.80 | Frame extraction, webcam, image ops |
| **Face Detection** | MediaPipe | 0.10.14 | Real-time face landmark detection |
| **Model Architecture** | XceptionNet | — | Transfer learning backbone |
| **Explainability** | Grad-CAM | — | Visual explanation heatmaps |
| **Visualization** | Plotly | 5.22.0 | Interactive charts and gauges |
| **Visualization** | Matplotlib | 3.8.4 | Heatmap rendering |
| **Visualization** | Seaborn | 0.13.2 | Statistical plotting |
| **Data Processing** | NumPy | 1.26.4 | Array & matrix operations |
| **Data Processing** | Pandas | 2.2.2 | Tabular data handling |
| **ML Utilities** | Scikit-learn | 1.4.2 | Metrics, calibration, splitting |
| **Image Processing** | Pillow | 10.3.0 | Image I/O and manipulation |
| **Augmentation** | Albumentations | 1.4.7 | Training data augmentation |
| **PDF Reports** | ReportLab | 4.2.0 | Professional PDF generation |
| **Database** | SQLite3 | stdlib | Detection history persistence |
| **Configuration** | PyYAML | 6.0.1 | Config file management |
| **Environment** | python-dotenv | 1.0.1 | Environment variable management |
| **Progress** | tqdm | 4.66.4 | CLI progress bars |
| **Dataset Access** | Kaggle API | 1.6.12 | Dataset downloading |

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 📁 Project Structure

<details open>
<summary><b>Click to expand</b></summary>

```
DFDC/
└── deepfake_detection/
    │
    ├── 📄 app.py                        # Main Streamlit application entry point
    ├── 📄 train.py                      # Model training entry point
    ├── 📄 requirements.txt              # Python dependencies
    ├── 📄 run.bat                       # Windows quick-launch script
    ├── 📄 README.md                     # Project documentation
    │
    ├── 📁 dashboard/                    # Streamlit UI layer
    │   ├── 📁 pages/                    # Individual dashboard pages
    │   │   ├── home.py                  # Landing page with KPI overview
    │   │   ├── image_detection.py       # Image upload & detection
    │   │   ├── video_detection.py       # Video upload & frame analysis
    │   │   ├── webcam_detection.py      # Live webcam detection
    │   │   ├── batch_detection.py       # Multi-file batch processing
    │   │   ├── reports.py               # PDF/CSV report generation
    │   │   ├── history.py               # Detection history & search
    │   │   ├── analytics.py             # Analytics dashboard
    │   │   ├── settings.py              # App configuration
    │   │   └── about.py                 # Project info
    │   ├── 📁 components/               # Reusable UI components
    │   │   ├── kpi_cards.py             # Animated KPI metric cards
    │   │   ├── charts.py                # Plotly chart components
    │   │   ├── gradcam_viewer.py        # Grad-CAM heatmap viewer
    │   │   ├── loader.py                # Animated loading components
    │   │   └── sidebar.py               # Navigation sidebar
    │   └── 📁 static/                   # Static assets
    │       └── style.css                # Global CSS (glassmorphism theme)
    │
    ├── 📁 inference/                    # Core detection engine
    │   ├── image_detector.py            # Image detection pipeline
    │   └── video_detector.py            # Video detection pipeline
    │
    ├── 📁 models/                       # Model architecture definitions
    │   ├── xceptionnet.py               # XceptionNet architecture
    │   ├── efficientnet.py              # EfficientNet (alternative)
    │   ├── resnet50.py                  # ResNet50 (alternative)
    │   └── model_factory.py             # Model loading & factory
    │
    ├── 📁 gradcam/                      # Explainability module
    │   └── grad_cam.py                  # Grad-CAM implementation
    │
    ├── 📁 preprocessing/                # Data preprocessing pipeline
    │   ├── face_extractor.py            # Face detection & extraction
    │   ├── pipeline.py                  # Full preprocessing pipeline
    │   ├── augmentor.py                 # Data augmentation
    │   └── dataset_validator.py         # Dataset integrity validation
    │
    ├── 📁 training/                     # Model training modules
    │   ├── trainer.py                   # Standard training loop
    │   ├── progressive_trainer.py       # Progressive unfreezing trainer
    │   └── data_loader.py               # TF Dataset loading
    │
    ├── 📁 evaluation/                   # Model evaluation
    │   ├── evaluator.py                 # Evaluation orchestrator
    │   └── metrics.py                   # Accuracy, F1, AUC, ROC metrics
    │
    ├── 📁 webcam/                       # Real-time webcam detection
    │   └── webcam_detector.py           # OpenCV webcam capture & inference
    │
    ├── 📁 reports/                      # Report generation
    │   └── pdf_generator.py             # ReportLab PDF compiler
    │
    ├── 📁 database/                     # Data persistence layer
    │   ├── schema.py                    # SQLite schema definition
    │   └── repository.py                # CRUD operations & queries
    │
    ├── 📁 configs/                      # Configuration files
    ├── 📁 assets/                       # Static assets & diagrams
    │   └── diagrams/                    # Workflow diagram images
    ├── 📁 data/                         # Raw & processed datasets
    ├── 📁 outputs/                      # Model checkpoints & exports
    ├── 📁 logs/                         # Training & application logs
    ├── 📁 scripts/                      # Utility scripts
    └── 📁 tests/                        # Unit & integration tests
```

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 🚀 Installation Guide

<details open>
<summary><b>Click to expand</b></summary>

### Prerequisites

- Python **3.11.x** (recommended; `python --version` to check)
- pip **23+** or `conda`
- Webcam (optional, for live detection)
- GPU with CUDA (optional, for faster training; CPU supported)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection/deepfake_detection
```

---

### Step 2 — Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** All dependencies are pinned in `requirements.txt` for reproducibility. TensorFlow 2.15.1 runs on CPU by default; GPU support requires a compatible CUDA installation.

---

### Step 4 — Run the Application

**Using the launcher script (Windows):**
```powershell
.\run.bat
```

**Using Streamlit directly:**
```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`

---

### Step 5 — (Optional) Train the Model

```bash
python train.py
```

> Configure training parameters in `configs/` before running. Pre-trained weights can be placed in `outputs/` and will be auto-loaded by the model factory.

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 📦 Datasets

<details>
<summary><b>Click to expand</b></summary>

This project was designed and evaluated against the three most widely used and rigorously curated DeepFake datasets in academic research:

| Dataset | Publisher | Size | Manipulation Types | Access |
|---------|-----------|------|--------------------|--------|
| **FaceForensics++** | TU Munich | ~1.5M frames | Face swap, face reenactment, neural textures | [Request Access](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF** | Stevens Institute | ~590K frames | High-quality celebrity face swaps | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC** (DeepFake Detection Challenge) | Meta AI / Kaggle | ~119K videos | Multi-modal, diverse demographics | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |

### Why These Datasets?

- **FaceForensics++** — Gold-standard benchmark with multiple forgery methods at varying compression levels; enables controlled evaluation across manipulation types.
- **Celeb-DF** — High-quality visual fidelity that challenges detectors trained on lower-quality fakes; tests generalization.
- **DFDC** — Largest and most demographically diverse dataset; created by Meta to specifically stress-test detection systems at production scale.

### Data Pipeline

```
Raw Videos → Frame Extraction → Face Detection (MediaPipe) 
          → Face Alignment → Crop & Resize (224×224)
          → Normalize → Train/Val/Test Split (70/15/15)
```

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 🧠 Model Details

<details open>
<summary><b>Click to expand</b></summary>

### Architecture: XceptionNet

**Xception** (Extreme Inception) is a convolutional neural network architecture introduced by François Chollet (Google) in 2017. It replaces standard convolutions with **depthwise separable convolutions**, dramatically improving parameter efficiency while maintaining representational power.

| Parameter | Value |
|-----------|-------|
| Base Architecture | Xception (ImageNet pretrained) |
| Input Shape | 224 × 224 × 3 |
| Backbone Layers | 71 layers (frozen during phase 1) |
| Classification Head | GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → Dense(1, Sigmoid) |
| Task | Binary Classification (Real / Fake) |
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam (lr=1e-4, phase 1) / SGD (lr=1e-5, phase 2) |
| Output | Probability ∈ [0, 1] — Fake likelihood |

### Transfer Learning Strategy

Training proceeds in two phases using **progressive unfreezing**:

```
Phase 1 (Feature Extraction)
  ├── Backbone: FROZEN (ImageNet weights preserved)
  ├── Head: TRAINED (task-specific layers)
  └── Epochs: 10-20, LR: 1e-4

Phase 2 (Fine-Tuning)
  ├── Backbone: PARTIALLY UNFROZEN (top layers)
  ├── Head: CONTINUED TRAINING
  └── Epochs: 5-10, LR: 1e-5 (reduced to prevent catastrophic forgetting)
```

### Explainability: Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** computes the gradient of the class prediction score with respect to the final convolutional feature map. The resulting heatmap localizes which regions most contributed to the model's decision.

```
Grad-CAM Algorithm:
  1. Forward pass → obtain class score ŷ
  2. Backward pass → compute ∂ŷ/∂Aᵏ for each feature map Aᵏ
  3. Global Average Pool the gradients → αᵏ (importance weights)
  4. Weighted combination: Lcam = ReLU(Σ αᵏ · Aᵏ)
  5. Upsample to input resolution
  6. Overlay as heatmap (jet colormap) on original image
```

**Interpretation:**
- 🔴 **Hot regions (red/yellow)** — Areas driving the "FAKE" prediction (e.g., eye boundaries, chin edges, blending artifacts)
- 🔵 **Cool regions (blue/dark)** — Areas with low influence on the decision

### Confidence Score (Mathematically Consistent)

The system enforces strict mathematical consistency:

```
fake_prob   = model_output         ∈ [0, 1]
real_prob   = 1 - fake_prob        ∈ [0, 1]
prediction  = "FAKE" if fake_prob > threshold else "REAL"
confidence  = prob_of_predicted_class × 100%

# Invariants enforced:
# ✅ fake_prob + real_prob == 1.0
# ✅ confidence == max(fake_prob, real_prob) × 100
# ✅ Never: "Prediction: REAL, Real Probability: 9%"
```

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 📸 Dashboard Preview

> **📌 Note for maintainer:** Replace the placeholder sections below with actual application screenshots when available. Use the format shown for consistency.

<details open>
<summary><b>Workflow Diagrams</b></summary>

<br/>

<div align="center">

| Pipeline | Diagram |
|----------|---------|
| End-to-End System | ![System Workflow](assets/diagrams/workflow_overview.png) |
| Training Pipeline | ![Training Pipeline](assets/diagrams/training_pipeline.png) |
| Batch Processing | ![Batch Workflow](assets/diagrams/batch_workflow.png) |
| Report Generation | ![Report Workflow](assets/diagrams/report_workflow.png) |
| Model Evaluation | ![Metrics Pipeline](assets/diagrams/metrics_pipeline.png) |

</div>

</details>

<details>
<summary><b>Application Screenshots</b></summary>

<br/>

---

#### 🏠 Home Dashboard

<!-- SCREENSHOT PLACEHOLDER: Home Dashboard -->
<!-- Insert: A full-width screenshot of the home/landing page showing KPI cards, session stats, and navigation. -->
<!-- File naming convention: assets/screenshots/home_dashboard.png -->
<!-- Caption: "Home Dashboard — KPI overview, session statistics, and navigation sidebar" -->

```
[ HOME DASHBOARD SCREENSHOT ]
File: assets/screenshots/home_dashboard.png
Caption: Home Dashboard — Real-time KPI cards and session overview
```

---

#### 🖼️ Image Detection

<!-- SCREENSHOT PLACEHOLDER: Image Detection Page -->
<!-- Insert: Screenshot showing an uploaded image, face bounding box, prediction badge (REAL/FAKE), confidence gauge, and Grad-CAM overlay. -->
<!-- File naming convention: assets/screenshots/image_detection.png -->

```
[ IMAGE DETECTION SCREENSHOT ]
File: assets/screenshots/image_detection.png
Caption: Image Detection — Uploaded image, prediction badge, confidence score, and Grad-CAM heatmap
```

---

#### 🎬 Video Detection

<!-- SCREENSHOT PLACEHOLDER: Video Detection Page -->
<!-- Insert: Screenshot showing frame-wise analysis timeline, overall verdict, frame gallery with individual confidences. -->
<!-- File naming convention: assets/screenshots/video_detection.png -->

```
[ VIDEO DETECTION SCREENSHOT ]
File: assets/screenshots/video_detection.png
Caption: Video Detection — Frame-wise analysis timeline and aggregate verdict
```

---

#### 🔥 Grad-CAM Explainability

<!-- SCREENSHOT PLACEHOLDER: Grad-CAM Visualization -->
<!-- Insert: Side-by-side comparison of original face and Grad-CAM heatmap overlay. -->
<!-- File naming convention: assets/screenshots/gradcam_visualization.png -->

```
[ GRAD-CAM SCREENSHOT ]
File: assets/screenshots/gradcam_visualization.png
Caption: Grad-CAM Explainability — Heatmap overlay identifying manipulated facial regions
```

---

#### 📷 Webcam Detection

<!-- SCREENSHOT PLACEHOLDER: Webcam Detection Page -->
<!-- Insert: Screenshot of live webcam feed with bounding box, real-time confidence score, FPS counter, and REAL/FAKE badge. -->
<!-- File naming convention: assets/screenshots/webcam_detection.png -->

```
[ WEBCAM DETECTION SCREENSHOT ]
File: assets/screenshots/webcam_detection.png
Caption: Webcam Detection — Live real-time detection with FPS display and confidence overlay
```

---

#### 📦 Batch Detection

<!-- SCREENSHOT PLACEHOLDER: Batch Detection Page -->
<!-- Insert: Screenshot showing multi-file upload queue, progress bar, results table with verdicts, and CSV export button. -->
<!-- File naming convention: assets/screenshots/batch_detection.png -->

```
[ BATCH DETECTION SCREENSHOT ]
File: assets/screenshots/batch_detection.png
Caption: Batch Detection — Multi-file processing queue with aggregated results and CSV export
```

---

#### 📄 PDF Reports

<!-- SCREENSHOT PLACEHOLDER: Reports Page -->
<!-- Insert: Screenshot of the reports page showing detection history list, generate/download PDF buttons, and report preview. -->
<!-- File naming convention: assets/screenshots/reports_page.png -->

```
[ REPORTS SCREENSHOT ]
File: assets/screenshots/reports_page.png
Caption: Report Generation — Automated PDF reports with detection evidence and Grad-CAM
```

---

#### 🕘 Detection History

<!-- SCREENSHOT PLACEHOLDER: Detection History Page -->
<!-- Insert: Screenshot of history page with searchable table, filter dropdowns, date range selector, and export controls. -->
<!-- File naming convention: assets/screenshots/detection_history.png -->

```
[ DETECTION HISTORY SCREENSHOT ]
File: assets/screenshots/detection_history.png
Caption: Detection History — SQLite-backed searchable and filterable detection log
```

---

#### 📊 Analytics Dashboard

<!-- SCREENSHOT PLACEHOLDER: Analytics Page -->
<!-- Insert: Screenshot showing detection trend charts, real/fake ratio donut, confidence distribution histogram, and session KPIs. -->
<!-- File naming convention: assets/screenshots/analytics_dashboard.png -->

```
[ ANALYTICS DASHBOARD SCREENSHOT ]
File: assets/screenshots/analytics_dashboard.png
Caption: Analytics Dashboard — Detection trends, confidence distributions, and session KPIs
```

---

> **To add screenshots:**
> 1. Take screenshots of the running application
> 2. Save them in `assets/screenshots/` with the filenames shown above
> 3. Replace the code blocks above with: `![Caption](assets/screenshots/filename.png)`

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 📊 Performance Metrics

<details open>
<summary><b>Click to expand</b></summary>

> [!NOTE]
> Performance metrics below reflect results from the model evaluation pipeline. Final production metrics depend on the training dataset, augmentation strategy, and hardware used. Values marked with ⏳ will be updated after full production training.

### Model Performance

<div align="center">

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | ⏳ To be updated after production training | Overall correct classifications |
| **Precision** | ⏳ To be updated after production training | TP / (TP + FP) |
| **Recall** | ⏳ To be updated after production training | TP / (TP + FN) |
| **F1 Score** | ⏳ To be updated after production training | Harmonic mean of Precision & Recall |
| **ROC-AUC** | ⏳ To be updated after production training | Area under ROC curve |
| **Confusion Matrix** | ⏳ To be updated after production training | TP / FP / TN / FN breakdown |

</div>

### Evaluation Methodology

```
Dataset Split:      Train 70% / Validation 15% / Test 15%
Evaluation Data:    Held-out test set (never seen during training)
Threshold:          0.5 (configurable in Settings)
Metrics Computed:   Accuracy, Precision, Recall, F1, AUC, ROC Curve
```

### Performance Visualization

<div align="center">
<img src="assets/diagrams/metrics_pipeline.png" alt="Model Evaluation Pipeline" width="90%"/>
<br/><em>Model Evaluation Pipeline — Confusion Matrix · ROC Curve · AUC Score</em>
</div>

<!-- METRICS SCREENSHOT PLACEHOLDER -->
<!-- After production training, insert: -->
<!-- - ROC Curve plot (assets/screenshots/roc_curve.png) -->
<!-- - Confusion Matrix (assets/screenshots/confusion_matrix.png) -->
<!-- - Training/Validation loss curves (assets/screenshots/training_curves.png) -->

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 🔮 Future Enhancements

<details>
<summary><b>Click to expand</b></summary>

| Priority | Enhancement | Description |
|----------|------------|-------------|
| 🔴 High | **Cloud Deployment** | Deploy on AWS / GCP / Azure with auto-scaling and load balancing |
| 🔴 High | **REST API** | FastAPI/Flask wrapper exposing `/detect/image`, `/detect/video` endpoints |
| 🟡 Medium | **Mobile Application** | React Native app with on-device TFLite inference |
| 🟡 Medium | **Multi-Face Detection** | Simultaneous detection of multiple faces in a single frame |
| 🟡 Medium | **ONNX Export** | Export to ONNX for cross-platform, framework-agnostic inference |
| 🟡 Medium | **GPU Optimization** | CUDA-optimized inference with TensorRT and mixed-precision (FP16) |
| 🟢 Low | **Improved Training Data** | Expand training with DFDC, WildDeepfake, and FaceShifter datasets |
| 🟢 Low | **Audio DeepFake Detection** | Extend pipeline to detect AI-synthesized voice cloning |
| 🟢 Low | **Real-time Stream Analysis** | RTSP/HLS stream ingestion for broadcast media monitoring |
| 🟢 Low | **Transformer Architecture** | Experiment with ViT, CLIP, and hybrid CNN-Transformer models |
| 🟢 Low | **Browser Extension** | Chrome/Firefox extension for on-the-fly web media verification |

</details>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

See the full [LICENSE](LICENSE) file for details.

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 👤 Author

<div align="center">

### Built with ❤️ by

<br/>

<!-- Replace the placeholders below with your actual information -->

| | |
|-|-|
| **Name** | `[Your Full Name]` |
| **Degree** | B.Tech — `[Your Branch]`, `[Your College/University]` |
| **Batch** | `[Year of Graduation]` |
| 🐙 **GitHub** | [github.com/your-username](https://github.com/your-username) |
| 💼 **LinkedIn** | [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile) |
| 📧 **Email** | `your.email@example.com` |
| 🌐 **Portfolio** | `your-portfolio-website.com` *(optional)* |

<br/>

> *This project was developed as a Final Year B.Tech capstone project in the domain of Computer Vision and Explainable AI.*

</div>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>

---

## 🙏 Acknowledgements

This project builds upon the outstanding work of the following organizations, research teams, and open-source communities:

| Resource | Contribution |
|----------|-------------|
| 🔶 **TensorFlow / Google Brain** | Deep learning framework and pre-trained XceptionNet weights |
| 👁️ **OpenCV** | Computer vision primitives, video I/O, and webcam capture |
| 🌊 **Streamlit** | Rapid interactive web dashboard development |
| 🎯 **MediaPipe (Google)** | Real-time face detection and landmark localization |
| 📊 **Plotly** | Interactive charting and visualization components |
| 📑 **ReportLab** | Professional PDF generation engine |
| 🔬 **FaceForensics++ (TU Munich)** | Benchmark dataset and evaluation protocol |
| ⭐ **Celeb-DF (Stevens Institute)** | High-quality DeepFake evaluation dataset |
| 🏆 **DFDC (Meta AI)** | Large-scale diverse DeepFake detection challenge dataset |
| 🧬 **François Chollet** | Original Xception architecture paper and Keras framework |
| 📖 **Grad-CAM Authors** (Selvaraju et al.) | *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, ICCV 2017 |
| 🤗 **Open-Source Community** | NumPy, Pandas, Scikit-learn, Albumentations, and all dependent libraries |

---

<div align="center">

**⭐ If this project was helpful, please consider starring the repository!**

[![GitHub Stars](https://img.shields.io/github/stars/your-username/deepfake-detection?style=social)](https://github.com/your-username/deepfake-detection)
[![GitHub Forks](https://img.shields.io/github/forks/your-username/deepfake-detection?style=social)](https://github.com/your-username/deepfake-detection/fork)

<br/>

*Made with 🧠 + ☕ | Final Year B.Tech Project | 2026*

</div>

<p align="right"><a href="#-table-of-contents">↑ Back to Top</a></p>
