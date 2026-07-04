# Application Screenshots — Expected File Names

Place your application screenshots in this directory (`assets/screenshots/`) with EXACTLY these filenames
to have them automatically appear in the README.md Dashboard Preview section.

## Required Screenshots

| Filename | Page/Feature | What to Capture |
|----------|-------------|----------------|
| `home_dashboard.png` | Home Page | KPI cards, session stats, navigation sidebar |
| `image_detection.png` | Image Detection | Uploaded image, prediction badge, confidence gauge, Grad-CAM overlay |
| `video_detection.png` | Video Detection | Frame-wise timeline, aggregate verdict, frame gallery |
| `gradcam_visualization.png` | Grad-CAM | Side-by-side original face vs. heatmap overlay |
| `webcam_detection.png` | Webcam | Live feed with bounding box, FPS, confidence badge |
| `batch_detection.png` | Batch Detection | Upload queue, progress, results table, CSV export |
| `reports_page.png` | Reports | History list, PDF generate/download, preview |
| `detection_history.png` | History | Searchable table, filter controls, export options |
| `analytics_dashboard.png` | Analytics | Trend charts, real/fake donut, confidence histogram |

## Optional / Bonus

| Filename | Page/Feature |
|----------|-------------|
| `roc_curve.png` | ROC Curve chart from evaluation |
| `confusion_matrix.png` | Confusion matrix visualization |
| `training_curves.png` | Training & validation loss/accuracy curves |
| `settings_page.png` | Settings & configuration panel |

## After Adding Screenshots

Update README.md — find each placeholder block like:
```
[ HOME DASHBOARD SCREENSHOT ]
File: assets/screenshots/home_dashboard.png
```
And replace it with:
```markdown
![Home Dashboard — KPI overview and session statistics](assets/screenshots/home_dashboard.png)
```
