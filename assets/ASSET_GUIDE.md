## 📁 Asset File Mapping Guide

This document tells you **exactly where to save every image** so GitHub renders them correctly in the README.

---

## ✅ Workflow Diagrams → `assets/workflows/`

Save each image with the **exact filename** shown below:

| README Section | Filename to Use | Description |
|----------------|----------------|-------------|
| End-to-End System Workflow | `system_workflow.png` | Full pipeline from input to results |
| Training Pipeline | `training_pipeline.png` | XceptionNet training with FF++, Celeb-DF, DFDC |
| Batch Detection Workflow | `batch_workflow.png` | Multi-file queue processing pipeline |
| Report Generation Workflow | `report_workflow.png` | PDF/CSV report automation flow |
| Performance Evaluation Pipeline | `metrics_pipeline.png` | Accuracy, F1, AUC, ROC evaluation |

**Path:** `deepfake_detection/assets/workflows/<filename>`

---

## ✅ Application Screenshots → `assets/screenshots/`

Save each screenshot with the **exact filename** shown below:

| README Section | Filename to Use | What to Capture |
|----------------|----------------|----------------|
| Home Dashboard | `home.png` | KPI cards, session stats, navigation |
| Image Detection | `image_detection.png` | Upload, verdict badge, Grad-CAM |
| Video Detection | `video_detection.png` | Frame timeline, gallery, verdict |
| Grad-CAM Explainability | `gradcam.png` | Original vs. heatmap side-by-side |
| Webcam Detection | `webcam.png` | Live feed, bounding box, badge |
| Batch Detection | `batch_detection.png` | Queue, progress, results table |
| PDF Reports | `reports.png` | Report list, generate/download |
| Detection History | `history.png` | Searchable table, filters |
| Analytics Dashboard | `analytics.png` | Charts, donut, trend lines |

**Path:** `deepfake_detection/assets/screenshots/<filename>`

---

## How the README Activates Images

The README uses this HTML tag format:

```html
<p align="center">
  <img src="assets/workflows/system_workflow.png" width="960" alt="System Workflow"/>
</p>
```

Once the image file exists at that path, GitHub renders it automatically. The `⚠️ Screenshot will be added` notice disappears as soon as you commit the image.

---

## Workflow — Adding New Images

1. Save your image to the correct folder with the correct filename
2. `git add assets/`
3. `git commit -m "Add workflow diagrams and screenshots"`
4. `git push`
5. Refresh your GitHub README — images appear instantly ✅
