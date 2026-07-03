"""
app.py
=======
Streamlit application entrypoint for the DeepFake Detection System.

Run with:
    streamlit run app.py

This module:
- Loads the CSS theme
- Initialises the database
- Loads the AI model (or runs in demo mode if no weights exist)
- Initialises all detectors
- Renders the selected page from the sidebar navigation
"""

import sys
import os
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Change working directory to project root so relative paths work
os.chdir(ROOT)

import streamlit as st
from utils.logger import get_logger, setup_logging
from utils.file_utils import load_yaml_config, ensure_dir

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging(log_dir="logs")
logger = get_logger(__name__)

# ── Streamlit page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "Explainable DeepFake Detection System v1.0",
    },
)


# ── Load CSS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    css_path = ROOT / "dashboard" / "static" / "style.css"
    if css_path.is_file():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# ── Config ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_config() -> dict:
    """Load the central YAML configuration (cached)."""
    return load_yaml_config(ROOT / "configs" / "config.yaml")


cfg = load_config()


# ── Database ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_repository():
    """Initialise and return the detection repository (cached)."""
    from database.repository import DetectionRepository
    db_path = ROOT / cfg["paths"]["database"]
    return DetectionRepository(db_path=db_path)


repo = get_repository()


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading AI model…")
def load_model_and_detectors():
    """
    Load the best available model and initialise all detectors.

    Falls back to a demo stub model (randomly initialised weights) if no
    trained checkpoint is found. The stub allows full dashboard exploration
    without training.
    """
    from models.model_factory import ModelFactory
    from inference.image_detector import ImageDetector
    from inference.video_detector import VideoDetector
    from gradcam.grad_cam import GradCAM
    from reports.pdf_generator import PDFReportGenerator

    model_cfg = cfg.get("models", {})
    infer_cfg = cfg.get("inference", {})
    paths_cfg = cfg.get("paths", {})
    gradcam_cfg = cfg.get("gradcam", {})

    weights_dir = ROOT / paths_cfg.get("weights", "outputs/weights")
    model_name_key = model_cfg.get("default", "xceptionnet")
    input_shape = tuple(model_cfg.get("input_shape", [299, 299, 3]))
    input_size = (input_shape[0], input_shape[1])
    threshold = infer_cfg.get("confidence_threshold", 0.5)

    # ── Try to load saved weights ─────────────────────────────────────────────
    model = None
    model_display_name = model_name_key.replace("_", " ").title()
    model_status = "⚠️ Demo Mode (no trained weights)"

    for candidate in [
        weights_dir / "best_model.keras",
        weights_dir / f"{model_name_key}_final.keras",
        weights_dir / f"{model_name_key}_phase_3_best.keras",
        weights_dir / f"{model_name_key}_phase_2_best.keras",
        weights_dir / f"{model_name_key}_phase_1_best.keras",
        weights_dir / "xceptionnet_final.keras",
        weights_dir / "xceptionnet_phase_1_best.keras",
    ]:
        if candidate.is_file():
            try:
                model = ModelFactory.load(candidate, compile_model=False)
                model_display_name = candidate.stem.replace("_", " ").title()
                model_status = f"✅ Loaded: `{candidate.name}`"
                logger.info("Model loaded from: %s", candidate)
                break
            except Exception as exc:
                logger.warning("Failed to load %s: %s", candidate, exc)

    # ── Demo stub (no weights found) ──────────────────────────────────────────
    if model is None:
        error_msg = (
            f"No trained model checkpoint found in {weights_dir}. "
            "Randomly initialized weights are never permitted in production. "
            "Please train the model first by running: "
            "python scripts/quick_train.py --data data/toy_dataset"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_layer = gradcam_cfg.get(
        f"target_layer_{model_name_key}",
        ModelFactory.get_gradcam_layer(model_name_key),
    )
    gradcam = None
    try:
        gradcam = GradCAM(model=model, target_layer=gradcam_layer)
    except Exception as exc:
        logger.warning("Grad-CAM initialisation failed: %s", exc)

    # ── Detectors ─────────────────────────────────────────────────────────────
    image_detector = ImageDetector(
        model=model,
        model_name=model_display_name,
        input_size=input_size,
        threshold=threshold,
        weights_dir=weights_dir,   # ← auto-loads calibration.json + threshold.json
    )

    video_detector = VideoDetector(
        image_detector=image_detector,
        sample_rate=infer_cfg.get("video_frame_sample_rate", 5),
        max_frames=infer_cfg.get("max_video_frames", 200),
        temporal_window=infer_cfg.get("temporal_window", 5),
        threshold=None,   # ← inherit tuned threshold from ImageDetector
    )

    # ── PDF generator ─────────────────────────────────────────────────────────
    pdf_gen = PDFReportGenerator(
        output_dir=str(ROOT / paths_cfg.get("reports", "outputs/reports"))
    )

    return {
        "model": model,
        "model_name": model_display_name,
        "model_status": model_status,
        "image_detector": image_detector,
        "video_detector": video_detector,
        "gradcam": gradcam,
        "pdf_gen": pdf_gen,
    }


resources = load_model_and_detectors()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2.5rem;">🛡️</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:800;
                        font-size:1.1rem; color:#5dade2; margin-top:0.3rem;">
                DeepFake Detector
            </div>
            <div style="font-size:0.7rem; color:#7f8c8d;">v1.0 · XceptionNet</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    nav_options = {
        "🏠 Home":              "home",
        "🖼️ Image Detection":   "image",
        "🎬 Video Detection":   "video",
        "📡 Webcam Detection":  "webcam",
        "📦 Batch Detection":   "batch",
        "📄 Reports":           "reports",
        "🕒 Detection History": "history",
        "ℹ️ About":             "about",
    }

    selected_page = st.radio(
        "Navigation",
        list(nav_options.keys()),
        label_visibility="collapsed",
        key="main_nav",
    )

    st.divider()

    # Model status
    st.markdown(
        f"""
        <div style="background:rgba(30,39,56,0.7); border-radius:8px;
                    padding:0.8rem; font-size:0.78rem;">
            <div style="color:#95a5a6; margin-bottom:0.3rem;">🤖 Model Status</div>
            <div style="color:#ecf0f1;">{resources['model_status']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page routing ──────────────────────────────────────────────────────────────
page_key = nav_options[selected_page]

if page_key == "home":
    from dashboard.pages.home import render
    render(repo=repo)

elif page_key == "image":
    from dashboard.pages.image_detection import render
    render(
        image_detector=resources["image_detector"],
        gradcam=resources["gradcam"],
        pdf_gen=resources["pdf_gen"],
        repo=repo,
        model_name=resources["model_name"],
    )

elif page_key == "video":
    from dashboard.pages.video_detection import render
    render(
        video_detector=resources["video_detector"],
        pdf_gen=resources["pdf_gen"],
        repo=repo,
        model_name=resources["model_name"],
    )

elif page_key == "webcam":
    from dashboard.pages.webcam_detection import render
    render(
        image_detector=resources["image_detector"],
        repo=repo,
        model_name=resources["model_name"],
    )

elif page_key == "batch":
    from dashboard.pages.batch_detection import render
    render(
        image_detector=resources["image_detector"],
        video_detector=resources["video_detector"],
        repo=repo,
        model_name=resources["model_name"],
    )

elif page_key == "reports":
    from dashboard.pages.reports import render
    render(reports_dir=str(ROOT / "outputs" / "reports"))

elif page_key == "history":
    from dashboard.pages.history import render
    render(repo=repo)

elif page_key == "about":
    from dashboard.pages.about import render as render_about
    render_about()
