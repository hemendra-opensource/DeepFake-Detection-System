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
            # except Exception as exc:
            #      print("=" * 80)
            #      print("MODEL LOAD ERROR")
            #      print(candidate)
            #      print(exc)
            #      import traceback
            #      traceback.print_exc()
            #      print("=" * 80)
            #      logger.warning("Failed to load %s: %s", candidate, exc)

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
        <div style="text-align:center; padding:1.2rem 0 0.5rem;">
            <div style="font-size:2.8rem; line-height: 1;">🛡️</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:900;
                        font-size:1.2rem; color:#3B82F6; margin-top:0.4rem;
                        letter-spacing:-0.01em;">
                DeepFake Detection AI
            </div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.2rem;">
                Version <b>v1.0</b> &nbsp;|&nbsp; Model <b>XceptionNet</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    nav_options = {
        "🏠 Dashboard":         "home",
        "🖼️ Image Detection":   "image",
        "🎥 Video Detection":   "video",
        "📷 Webcam Detection":  "webcam",
        "📦 Batch Detection":   "batch",
        "📊 Analytics":         "analytics",
        "📄 Reports":           "reports",
        "🕒 History":           "history",
        "⚙️ Settings":          "settings",
        "ℹ️ About":             "about",
    }

    selected_page = st.radio(
        "Navigation",
        list(nav_options.keys()),
        label_visibility="collapsed",
        key="main_nav",
    )

    st.divider()

    # ── Model Status ──────────────────────────────────────────────────────────
    is_loaded = resources['model_status'].startswith("✅")
    status_color = "#22C55E" if is_loaded else "#F59E0B"
    st.markdown(
        f"""
        <div style="background:var(--bg-card); border-radius:12px;
                    padding:0.9rem; font-size:0.82rem; border:1px solid var(--border);
                    border-left: 4px solid {status_color}; margin-bottom: 0.8rem;">
            <div style="color:var(--text-secondary); margin-bottom:0.3rem; font-weight:600;">
                🤖 Active Model
            </div>
            <div style="color:var(--text-primary); font-weight:700;">{resources['model_name']}</div>
            <div style="color:{status_color}; font-size:0.72rem; margin-top:0.2rem; font-weight:500;">
                {resources['model_status']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Resource meters ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:var(--bg-card); border-radius:12px;
                    padding:0.9rem; font-size:0.82rem; border:1px solid var(--border);
                    margin-bottom:0.8rem;">
            <div style="color:var(--text-secondary); font-weight:600; margin-bottom:0.4rem;">
                🖥️ System Status
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                <span>CPU / Memory</span><span style="font-weight:700;">Normal</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>GPU Acceleration</span><span style="font-weight:700; color:#F59E0B;">Not Available</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Theme Toggle ──────────────────────────────────────────────────────────
    theme_choice = st.selectbox(
        "Theme",
        ["🌙 Dark Theme", "☀️ Light Theme"],
        index=0 if st.session_state.get("theme", "dark") == "dark" else 1,
        key="theme_selector",
        label_visibility="collapsed",
    )
    if "Light" in theme_choice:
        st.session_state["theme"] = "light"
        st.markdown(
            """
            <style>
            :root {
              --bg-dark:        #F8FAFC !important;
              --bg-card:        rgba(255, 255, 255, 0.9) !important;
              --bg-card-hover:  rgba(59, 130, 246, 0.05) !important;
              --border:         rgba(59, 130, 246, 0.12) !important;
              --text-primary:   #0F172A !important;
              --text-secondary: #64748B !important;
              --glass:          rgba(0, 0, 0, 0.02) !important;
              --shadow:         0 10px 40px rgba(15, 23, 42, 0.06) !important;
            }
            .stApp {
              background: #F8FAFC !important;
              color: #0F172A !important;
            }
            [data-testid="stSidebar"] {
              background: #FFFFFF !important;
              border-right: 1px solid rgba(0,0,0,0.06) !important;
            }
            .stRadio > label, .stSelectbox > label, .stTextInput > label, .stSlider > label {
              color: #0F172A !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state["theme"] = "dark"


# ── Page routing ──────────────────────────────────────────────────────────────
page_key = nav_options[selected_page]

# Handle dashboard home page routes redirection helpers
if "home_redirect" in st.session_state:
    page_key = st.session_state.pop("home_redirect")
    # Reset radio selection visually
    for k, v in nav_options.items():
        if v == page_key:
            st.session_state["main_nav"] = k
            break

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
        gradcam=resources["gradcam"],
    )

elif page_key == "batch":
    from dashboard.pages.batch_detection import render
    render(
        image_detector=resources["image_detector"],
        video_detector=resources["video_detector"],
        repo=repo,
        model_name=resources["model_name"],
    )

elif page_key == "analytics":
    from dashboard.pages.analytics import render as render_analytics
    render_analytics(repo=repo)

elif page_key == "reports":
    from dashboard.pages.reports import render
    render(
        reports_dir=str(ROOT / "outputs" / "reports"),
        repo=repo,
        pdf_gen=resources["pdf_gen"],
    )

elif page_key == "history":
    from dashboard.pages.history import render
    render(repo=repo)

elif page_key == "settings":
    from dashboard.pages.settings import render as render_settings
    render_settings(
        image_detector=resources["image_detector"],
        video_detector=resources["video_detector"],
    )

elif page_key == "about":
    from dashboard.pages.about import render as render_about
    render_about()
