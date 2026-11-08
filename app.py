from __future__ import annotations



import base64

import io

import zipfile

from pathlib import Path



import numpy as np

import pandas as pd

import streamlit as st

from PIL import Image, ImageDraw, ImageFont



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

LOGO_PATH = BASE_DIR / "logo.png"

BACKGROUND_PATH = BASE_DIR / "background.png"

HERO_IMAGE_PATH = BASE_DIR / "pcb_hero.png"



APP_TITLE = "PCB AI Inspection Studio"

TAGLINE = "Precision · Detection · Inspection"

DEFAULT_CONFIDENCE = 0.25

SLICE_SIZE = 640

OVERLAP_RATIO = 0.20



st.set_page_config(

    page_title=APP_TITLE,

    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🔬",

    layout="wide",

    initial_sidebar_state="expanded",

)



# --------------------------- Styling ---------------------------

def image_as_data_uri(path: Path) -> str | None:

    if not path.is_file():

        return None

    try:

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")

        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

        return f"data:{mime};base64,{encoded}"

    except OSError:

        return None

def apply_styles() -> None:


    background_uri = image_as_data_uri(BACKGROUND_PATH)

    if background_uri:

        background_rule = f'background-image: linear-gradient(rgba(224,244,255,.28), rgba(224,244,255,.28)), url("{background_uri}");'

    else:

        background_rule = "background-image: radial-gradient(circle at 80% 15%, #d6f1ff 0, transparent 38%), linear-gradient(135deg, #eaf7ff, #c9eaff);"



    st.markdown(

        f"""

        <style>
        
        :root {{

          --navy: #102b49;

          --navy-2: #0b2038;

          --teal: #13c7d0;

          --teal-dark: #087f91;

          --ink: #123453;

          --muted: #45647f;

          --line: #b9e4f5;

        }}

        html, body, [data-testid="stApp"] {{

          {background_rule}

          background-size: cover !important;

          background-position: center center !important;

          background-attachment: fixed !important;

          background-repeat: no-repeat !important;

        }}

        [data-testid="stAppViewContainer"], [data-testid="stMain"],

        [data-testid="stMainBlockContainer"] {{ background: transparent !important; }}

        [data-testid="stHeader"] {{ background: rgba(226,245,255,.28) !important; }}

        [data-testid="stMainBlockContainer"] {{

          max-width: 1540px; padding-top: 1.35rem; padding-bottom: 2rem;

        }}

        [data-testid="stSidebar"] {{

          background: linear-gradient(180deg, #102d4c 0%, #0b2038 100%) !important;

          border-right: 1px solid rgba(36,211,221,.4);

        }}

        [data-testid="stSidebar"] * {{ color: #f2f8ff; }}

        [data-testid="stSidebar"] [data-testid="stRadio"] label {{

          border-radius: 10px; padding: .48rem .55rem; margin: .18rem 0;

        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{

          background: rgba(21,190,205,.17);

        }}

        .brand-mini {{ display:flex; align-items:center; gap:12px; padding: 8px 4px 16px; }}

        .brand-chip {{

          width:52px;height:52px;border:2px solid var(--teal);border-radius:12px;

          display:flex;align-items:center;justify-content:center;color:var(--teal);

          font-size:27px;box-shadow:0 0 16px rgba(19,199,208,.14);

        }}

        .brand-name {{font-weight:850;font-size:19px;line-height:1.1;letter-spacing:.2px;}}

        .brand-sub {{font-size:10px;letter-spacing:1.1px;font-weight:750;margin-top:5px;}}

        .brand-tagline {{color:#35d9de !important;font-size:12px;font-weight:700;margin-top:7px;}}

        .hero-panel {{
          position:relative; overflow:hidden; display:flex; align-items:center; gap:22px;
          min-height:150px; padding:24px 34px; margin:4px 0 26px;
          border-radius:16px; border:1px solid #17c9d1;
          background:linear-gradient(110deg, rgba(12,37,63,.98), rgba(14,49,77,.97)); /* <--- THIS IS BLOCKING IT */
          box-shadow:0 8px 24px rgba(12,49,77,.13);
        }}

        .hero-panel:after {{

          content:""; position:absolute; right:-30px; top:-70px; width:390px;height:300px;

          opacity:.26; background:

            repeating-linear-gradient(35deg, transparent 0 23px, #0aa6c3 24px 26px, transparent 27px 48px);

          transform:rotate(-8deg); pointer-events:none;

          mask-image:linear-gradient(90deg,transparent,black);

        }}

        .hero-logo {{

          width:88px;height:88px;object-fit:contain; flex-shrink:0;

          border:2px solid #26d6df;border-radius:14px;padding:5px;background:#ffffff;

        }}

        .hero-icon {{

          width:88px;height:88px;flex-shrink:0;border:2px solid #26d6df;border-radius:14px;

          display:flex;align-items:center;justify-content:center;font-size:48px;color:#28d8e0;

        }}

        .hero-panel h1 {{
    color: #37e1e2 !important;
    font-size: clamp(28px,3.2vw,42px) !important;
    font-weight: 850 !important;
}}


        .hero-kicker {{color:#fff;font-size:clamp(16px,1.8vw,23px);font-weight:800;letter-spacing:1px;}}

        .hero-tagline {{ 
    color: #37e1e2;
    font-size: 17px;
    font-weight: 700;
    margin-top: 10px;
    letter-spacing: .3px;
}}

        .section-heading {{display:flex;align-items:center;gap:12px;color:var(--ink);font-size:27px;font-weight:850;margin:0 0 2px;}}

        .section-note {{color:#385d79;font-size:14px;margin:2px 0 18px;}}

        /* Decorated dashboard cards */
.feature-card {{
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
    min-height: 205px;
    padding: 26px 28px;
    margin-bottom: 16px;

    background: linear-gradient(
        120deg,
        #0b2038 0%,
        #10435a 50%,
        #087f91 100%
    ) !important;

    border: 1px solid #24ccd5 !important;
    border-left: 5px solid #37e1e2 !important;
    border-radius: 18px;

    box-shadow: 0 8px 24px rgba(8, 45, 70, 0.22);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}}

/* Decorative ring in the top-right corner */
.feature-card::before {{
    content: "";
    position: absolute;
    top: -48px;
    right: -48px;
    width: 140px;
    height: 140px;
    border: 22px solid rgba(19, 199, 208, 0.10);
    border-radius: 50%;
    pointer-events: none;
}}

/* Short teal line above the title */
.feature-card::after {{
    content: "";
    position: absolute;
    top: 17px;
    left: 28px;
    width: 44px;
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #087f91, #37e1e2);
    pointer-events: none;
}}

.feature-card:hover {{
    transform: translateY(-4px);
    border-color: #13c7d0;
    box-shadow: 0 12px 28px rgba(8, 127, 145, 0.18);
}}

.feature-card h3 {{
    position: relative;
    z-index: 1;
    color: #082b50 !important;
    font-size: 21px;
    font-weight: 800;
    line-height: 1.35;
    margin: 12px 0 12px;
}}

.feature-card p {{
    position: relative;
    z-index: 1;
    color: #17324d !important;
    font-size: 14px;
    line-height: 1.75;
    margin: 0 0 8px;
}}

.feature-card strong {{
    color: #087f91;
}}

@media (prefers-reduced-motion: reduce) {{
    .feature-card {{
        transition: none;
    }}

    .feature-card:hover {{
        transform: none;
    }}
}}

        .feature-top {{display:flex;align-items:center;gap:12px;margin-bottom:10px;}}

        .feature-icon {{

          display:flex;align-items:center;justify-content:center;width:55px;height:55px;

          border-radius:12px;background:#e2f8ff;border:1px solid #b8eefa;color:#0784a4;font-size:27px;

        }}

        

        .mini-card {{

          background:rgba(255,255,255,.94);border:1px solid #b6e2f2;border-radius:12px;

          padding:17px 19px;min-height:140px;box-shadow:0 4px 13px rgba(20,89,127,.09);

        }}

        .mini-card h4 {{color:#102f4d !important;margin:9px 0 5px;font-size:16px;font-weight:800;}}

        .mini-card p {{color:#45647d !important;font-size:13px;line-height:1.55;margin:0;}}

        [data-testid="stMetric"] {{

          background:rgba(255,255,255,.95);border:1px solid #b9e4f5;border-radius:12px;padding:15px;

        }}

        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{color:#123453 !important;}}

        [data-testid="stFileUploader"] {{

          background:rgba(255,255,255,.91);border:1px solid #b8e3f3;border-radius:12px;padding:12px;

        }}

        div.stButton > button, div.stDownloadButton > button {{

          border-radius:9px;min-height:42px;font-weight:750;border:1px solid #087f91;

        }}

        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{

          background:#087f91;color:white;

        }}

        div.stButton > button[kind="primary"]:hover {{background:#076879;border-color:#076879;}}

        [data-testid="stDataFrame"] {{background:white;border:1px solid #b9e4f5;border-radius:10px;}}

        [data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] h4 {{color:#123453;}}

        [data-testid="stMain"] p {{color:#294d6a;}}

        .footer {{text-align:center;color:#355b79;font-size:11px;padding:22px 0 4px;}}

        hr {{border-color:rgba(70,157,190,.35) !important;}}

        @media(max-width:700px) {{
                  .hero-panel {{padding:20px;gap:14px;}}

          .hero-logo,.hero-icon {{width:62px;height:62px;}}

          .hero-tagline {{font-size:13px;}}

        }}

         
        /* Readable About section */
        .about-section {{
            background: #ffffff !important;
            color: #17324d !important;
            padding: 28px 32px;
            border-radius: 16px;
            border: 1px solid #9bdcf0;
            font-family: "Segoe UI", Arial, sans-serif;
            line-height: 1.8;
            box-shadow: 0 4px 18px rgba(20, 50, 80, 0.12);
        }}

        .about-section h2 {{
            color: #082b50 !important;
            font-size: 30px;
            font-weight: 800;
        }}

        .about-section h3 {{
            color: #087f91 !important;
            font-size: 22px;
            font-weight: 800;
        }}

        .about-section p,
        .about-section li {{
            color: #17324d !important;
            font-size: 16px !important;
            line-height: 1.8 !important;
        }}

        .about-section strong {{
            color: #006879 !important;
            font-weight: 800;
        }}

        /* Improve text contrast across the main page */
        [data-testid="stMain"] p,
        [data-testid="stMain"] li {{
            color: #17324d !important;
        }}

        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3 {{
            color: #082b50 !important;
        }}

       
        .feature-card h3 {{
            color: #082b50 !important;
        }}

        .feature-card p {{
            color: #17324d !important;
        }}
        /* Dark card background */
[data-testid="stMain"] .feature-card {{
    background: linear-gradient(
        120deg,
        #081b30 0%,
        #10384b 55%,
        #075360 100%
    ) !important;

    border: 1px solid #39cbd4 !important;
    border-left: 5px solid #37e1e2 !important;
}}

/* Readable card headings */
[data-testid="stMain"] .feature-card h3 {{
    position: relative;
    z-index: 1;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 23px !important;
    font-weight: 800 !important;
    line-height: 1.4;
    margin: 12px 0 14px;
    text-shadow: none;
    opacity: 1;
}}

/* Readable descriptions and list text */
[data-testid="stMain"] .feature-card p,
[data-testid="stMain"] .feature-card li {{
    position: relative;
    z-index: 1;
    color: #e8f4fa !important;
    -webkit-text-fill-color: #e8f4fa !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
    opacity: 1;
}}

/* Highlight labels within System Overview */
[data-testid="stMain"] .feature-card strong {{
    color: #72f1ed !important;
    -webkit-text-fill-color: #72f1ed !important;
    font-weight: 800 !important;
}}

/* Keep decorative elements subtle */
[data-testid="stMain"] .feature-card::before {{
    opacity: 0.25;
}}

/* Home-page module buttons */
.st-key-open_pcb button,
.st-key-open_gerber button {{
    background: linear-gradient(
        110deg,
        #075766 0%,
        #087f91 100%
    ) !important;

    color: #ffffff !important;
    border: 1px solid #37e1e2 !important;
    border-radius: 12px !important;

    min-height: 48px;
    font-size: 16px !important;
    font-weight: 750 !important;

    box-shadow: 0 4px 12px rgba(8, 127, 145, 0.20);
    transition: background 0.2s ease, box-shadow 0.2s ease;
}}

/* Override the global paragraph color inside these buttons */
[data-testid="stMain"] .st-key-open_pcb button p,
[data-testid="stMain"] .st-key-open_gerber button p {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 750 !important;
}}

/* Hover appearance */
.st-key-open_pcb button:hover,
.st-key-open_gerber button:hover {{
    background: #064b59 !important;
    border-color: #83f5ef !important;
    box-shadow: 0 6px 18px rgba(8, 127, 145, 0.30);
}}

/* Keyboard focus */
.st-key-open_pcb button:focus-visible,
.st-key-open_gerber button:focus-visible {{
    outline: 3px solid #087f91 !important;
    outline-offset: 3px;
}}

        </style>

        """,

        unsafe_allow_html=True,

    )

def apply_animations() -> None:
    st.html(
        """
<style>
/* Moving highlight across the hero */
@keyframes pcbStudioSweep {
    0% {
        left: -45%;
    }
    100% {
        left: 120%;
    }
}
@keyframes pcbCardGlow {
    0%, 100% {
        box-shadow:
            0 0 5px rgba(19, 199, 208, 0.20),
            0 6px 18px rgba(8, 45, 70, 0.20);
    }

    50% {
        box-shadow:
            0 0 20px rgba(19, 199, 208, 0.65),
            0 8px 24px rgba(8, 127, 145, 0.35);
    }
}

.hero-panel {
    position: relative !important;
    overflow: hidden !important;
}

/* Replace the old decorative pseudo-element */
.hero-panel::after {
    content: "" !important;
    display: block !important;
    position: absolute !important;

    top: 0 !important;
    bottom: auto !important;
    left: -45%;
    right: auto !important;

    width: 25% !important;
    height: 100% !important;

    background: linear-gradient(
        90deg,
        transparent,
        rgba(95, 245, 240, 0.55),
        transparent
    ) !important;

    opacity: 1 !important;
    transform: skewX(-18deg) !important;
    mask-image: none !important;
    -webkit-mask-image: none !important;
    pointer-events: none !important;
    z-index: 0 !important;

    animation: pcbStudioSweep 3s linear infinite !important;
}

/* Keep the logo and text above the highlight */
.hero-panel > * {
    position: relative;
    z-index: 1;
}

/* Clear older entrance animations from cards */
.feature-card,
.mini-card {
    animation: pcbCardGlow 3s ease-in-out infinite !important;
    transition: transform 250ms ease !important;
}

/* Buttons */
div.stButton > button,
div.stDownloadButton > button {
    transition:
        transform 200ms ease,
        box-shadow 200ms ease !important;
}

@media (hover: hover) and (pointer: fine) {
    .feature-card:hover,
    .mini-card:hover {
        transform: translateY(-6px) !important;
        box-shadow:
            0 14px 30px rgba(8, 127, 145, 0.35) !important;
    }

    div.stButton > button:not(:disabled):hover,
    div.stDownloadButton > button:not(:disabled):hover {
        transform: translateY(-3px) !important;
        box-shadow:
            0 8px 18px rgba(8, 127, 145, 0.30) !important;
    }
}

div.stButton > button:not(:disabled):active,
div.stDownloadButton > button:not(:disabled):active {
    transform: scale(0.97) !important;
}

/* Respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce) {
    .hero-panel::after {
        display: none !important;
        animation: none !important;
    }

    .feature-card,
.mini-card,
div.stButton > button,
div.stDownloadButton > button {
    animation: none !important;
    transition: none !important;
    transform: none !important;
}
}
</style>
"""
    )

apply_styles()
apply_animations()
@st.cache_data
def make_teal_title_uri() -> str:
    font_path = BASE_DIR / "DejaVuSans-Bold.ttf"

    try:
        # Prefer a font included in your GitHub repository.
        font = ImageFont.truetype(str(font_path), 96)
    except OSError:
        # Use Pillow's bundled default font if the file is missing.
        try:
            font = ImageFont.load_default(size=96)
        except TypeError:
            # Compatibility fallback for older Pillow versions.
            font = ImageFont.load_default()

    text = "PCB AI"
    padding = 8

    probe = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(probe)

    left, top, right, bottom = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    title = Image.new(
        "RGBA",
        (
            right - left + padding * 2,
            bottom - top + padding * 2,
        ),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(title)
    draw.text(
        (padding - left, padding - top),
        text,
        font=font,
        fill=(55, 225, 226, 255),
    )

    buffer = io.BytesIO()
    title.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def render_hero() -> None:
    logo = image_as_data_uri(LOGO_PATH)
    hero_bg = image_as_data_uri(HERO_IMAGE_PATH)
    title_uri = make_teal_title_uri()

    if logo:
        icon = (
            f'<img class="hero-logo" src="{logo}" '
            'alt="PCB AI logo">'
        )
    else:
        icon = '<div class="hero-icon">▦</div>'

    if hero_bg:
        background_style = (
            "background-image: "
            "linear-gradient("
            "90deg, "
            "rgba(4, 20, 40, 0.95) 0%, "
            "rgba(4, 20, 40, 0.82) 32%, "
            "rgba(4, 20, 40, 0.35) 65%, "
            "rgba(4, 20, 40, 0.08) 100%"
            "), "
            f"url('{hero_bg}');"
            "background-size: cover;"
            "background-position: center;"
            "background-repeat: no-repeat;"
        )
    else:
        background_style = (
            "background: linear-gradient("
            "90deg, #041428 0%, #08374b 50%, #087f91 100%"
            ");"
        )

    st.markdown(
        f"""
<div class="hero-panel" style="{background_style}">
    {icon}
    <div style="position:relative; z-index:1;">
       <img
    src="{title_uri}"
    alt="PCB AI"
    style="
        display:block;
        width:180px;
        max-width:100%;
        height:auto;
        margin:0 0 12px;
        opacity:1;
        filter:none;
    "
>
        <div class="hero-kicker"
             style="color:#ffffff !important;
                    text-shadow:0 1px 3px rgba(0,0,0,0.8);">
            INSPECTION STUDIO
        </div>
        <div class="hero-tagline"
             style="color:#37e1e2 !important;
                    font-weight:800;
                    text-shadow:0 1px 3px rgba(0,0,0,0.8);">
            Precision · Detection · Inspection
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
def render_footer() -> None:

    st.markdown('<div class="footer">PCB AI Inspection Studio &nbsp;·&nbsp; YOLO + SAHI &nbsp;·&nbsp; Gerber Analysis</div>', unsafe_allow_html=True)





apply_styles()



# ---------------------- PCB detection backend ----------------------

@st.cache_resource(show_spinner=False)

def load_yolo_model(model_path: str):

    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(

        model_type="ultralytics",

        model_path=model_path,

        confidence_threshold=0.25,

        device="cpu",

    )





def run_pcb_detection(image: Image.Image, confidence: float):

    from sahi.predict import get_sliced_prediction

    from sahi import AutoDetectionModel



    model = AutoDetectionModel.from_pretrained(

        model_type="ultralytics",

        model_path=str(MODEL_PATH),

        confidence_threshold=float(confidence),

        device="cpu",

    )

    return get_sliced_prediction(

        image=np.array(image.convert("RGB")),

        detection_model=model,

        slice_height=SLICE_SIZE,

        slice_width=SLICE_SIZE,

        overlap_height_ratio=OVERLAP_RATIO,

        overlap_width_ratio=OVERLAP_RATIO,

        perform_standard_pred=True,

        postprocess_type="GREEDYNMM",

        postprocess_match_metric="IOS",

        postprocess_match_threshold=0.5,

    ).object_prediction_list





def predictions_to_dataframe(predictions) -> pd.DataFrame:

    rows = []

    for idx, pred in enumerate(predictions, start=1):

        x1, y1, x2, y2 = pred.bbox.to_voc_bbox()

        rows.append({

            "Detection ID": idx,

            "Defect Class": pred.category.name,

            "Confidence (%)": round(float(pred.score.value) * 100, 2),

            "X1": round(float(x1), 2), "Y1": round(float(y1), 2),

            "X2": round(float(x2), 2), "Y2": round(float(y2), 2),

            "Width": round(float(x2 - x1), 2), "Height": round(float(y2 - y1), 2),

        })

    return pd.DataFrame(rows, columns=[

        "Detection ID", "Defect Class", "Confidence (%)", "X1", "Y1", "X2", "Y2", "Width", "Height"

    ])





def annotate_pcb_image(image: Image.Image, predictions) -> Image.Image:

    annotated = image.convert("RGB").copy()

    draw = ImageDraw.Draw(annotated)

    try:

        font = ImageFont.truetype("arial.ttf", 16)

    except OSError:

        font = ImageFont.load_default()

    for pred in predictions:

        x1, y1, x2, y2 = [int(round(float(v))) for v in pred.bbox.to_voc_bbox()]

        label = f"{pred.category.name} {float(pred.score.value) * 100:.1f}%"

        draw.rectangle([x1, y1, x2, y2], outline=(255, 35, 35), width=4)

        text_box = draw.textbbox((x1, max(0, y1 - 22)), label, font=font)

        draw.rectangle(text_box, fill=(220, 25, 25))

        draw.text((x1, max(0, y1 - 22)), label, fill=(255, 255, 255), font=font)

    return annotated





def render_pcb_page() -> None:

    st.markdown('<div class="section-heading">⌕ &nbsp; PCB Image Fault Detection</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-note">Upload a PCB image to identify trained visual defects with YOLO + SAHI sliced inference.</div>', unsafe_allow_html=True)

    if not MODEL_PATH.is_file():

        st.error(f"Trained model not found: {MODEL_PATH}. Place best.pt in the same folder as app.py.")

        return

    left, right = st.columns([1, 1])

    with left:

        confidence = st.slider("Detection confidence threshold", 0.05, 0.95, DEFAULT_CONFIDENCE, 0.05, help="Higher values filter out lower-confidence detections.")

    with right:

        st.info(f"SAHI slice: {SLICE_SIZE} × {SLICE_SIZE} px · Overlap: {OVERLAP_RATIO:.0%}")

    uploaded = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="pcb_image_upload")

    if uploaded is None:

        st.markdown(
    """
    <div class="feature-card">
        <h3 style="color: #082b50; margin-bottom: 12px;">How image inspection works</h3>
        <ol style="color: #385d79; font-size: 14px; padding-left: 20px; line-height: 1.6;">
            <li>Upload a clear PCB photograph.</li>
            <li>YOLOv8 + SAHI examines image regions.</li>
            <li>Detected classes appear in red boxes.</li>
            <li>Export detection details as a CSV report.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

        return

    try:

        original = Image.open(uploaded).convert("RGB")

    except Exception as exc:

        st.error(f"Could not open this image: {exc}")

        return

    st.image(original, caption="Uploaded PCB image", use_container_width=True)

    if st.button(
        "▶  Run PCB Fault Detection",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Loading model and inspecting PCB image…"):
                preds = run_pcb_detection(original, confidence)
                annotated = annotate_pcb_image(original, preds)

                st.session_state["pcb_result"] = {
                    "image": annotated,
                    "table": predictions_to_dataframe(preds),
                    "count": len(preds),
                    "source_name": uploaded.name,
                }

        except ImportError as exc:
            st.error(
                f"A required detection package is missing: {exc}. "
                "Install the dependencies in requirements.txt."
            )

        except Exception as exc:
            st.error(f"PCB detection failed: {exc}")

    result = st.session_state.get("pcb_result")

    if not result or result["source_name"] != uploaded.name:
        return

    st.divider()
    st.markdown("### Detection Results")

    a, b = st.columns(2)
    a.metric("Total Detected Objects", result["count"])
    b.metric(
        "Inspection Status",
        "Objects Detected" if result["count"] else "No Detections",
    )

    st.image(
        result["image"],
        caption="PCB image with detected defects highlighted in red",
        use_container_width=True,
    )

    if result["count"] == 0:
        st.info(
            "No objects were detected above the selected confidence threshold. "
            "This does not guarantee that the PCB is defect-free."
        )

    st.markdown("#### Detection Details")
    st.dataframe(
        result["table"],
        use_container_width=True,
        hide_index=True,
    )

    buf = io.BytesIO()
    result["image"].save(buf, format="PNG")

    c1, c2 = st.columns(2)

    c1.download_button(
        "⬇ Download Detection CSV",
        result["table"].to_csv(index=False).encode("utf-8"),
        "pcb_detection_report.csv",
        "text/csv",
        use_container_width=True,
    )

    c2.download_button(
        "⬇ Download Annotated PCB Image",
        buf.getvalue(),
        "pcb_annotated_result.png",
        "image/png",
        use_container_width=True,
    )





# ---------------------- Gerber inventory backend ----------------------

GERBER_EXTENSIONS = {

    ".gbr", ".ger", ".gerber", ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo",

    ".gtp", ".gbp", ".gko", ".gm1", ".gml", ".g2", ".g3", ".drl", ".xln", ".art",

    ".pho", ".cmp", ".sol", ".stc", ".sts", ".plc", ".pls", ".outline",

}

EXPECTED_LAYERS = {

    "Top Copper": ["gtl", "cmp", "top_copper", "top-copper", "copper_top", "top copper"],

    "Bottom Copper": ["gbl", "sol", "bottom_copper", "bottom-copper", "copper_bottom", "bottom copper"],

    "Top Solder Mask": ["gts", "stc", "top_mask", "top-mask", "top_soldermask", "top solder mask"],

    "Bottom Solder Mask": ["gbs", "sts", "bottom_mask", "bottom-mask", "bottom_soldermask", "bottom solder mask"],

    "Top Silkscreen": ["gto", "plc", "top_silkscreen", "top-silkscreen", "top legend"],

    "Bottom Silkscreen": ["gbo", "pls", "bottom_silkscreen", "bottom-silkscreen", "bottom legend"],

    "Board Outline / Mechanical": ["gko", "gm1", "gml", "outline", "board_outline", "board-outline", "edge_cuts", "profile"],

    "Drill Data": ["drl", "xln", "drill", "ncdrill", "excellon"],

}

EXTENSION_MAP = {

    "gtl": "Top Copper", "cmp": "Top Copper", "gbl": "Bottom Copper", "sol": "Bottom Copper",

    "gts": "Top Solder Mask", "stc": "Top Solder Mask", "gbs": "Bottom Solder Mask",

    "sts": "Bottom Solder Mask", "gto": "Top Silkscreen", "plc": "Top Silkscreen",

    "gbo": "Bottom Silkscreen", "pls": "Bottom Silkscreen", "gko": "Board Outline / Mechanical",

    "gm1": "Board Outline / Mechanical", "gml": "Board Outline / Mechanical", "drl": "Drill Data", "xln": "Drill Data",

}


def identify_layer(filename: str) -> str:
    path = Path(filename)

    # First, identify layers from common file extensions.
    ext = path.suffix.lower().lstrip(".")

    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]

    # Next, identify layers from KiCad-style Gerber filenames.
    name = path.stem.lower().replace("-", "_").replace(" ", "_")

    if "_f_cu" in name or name.endswith("f_cu"):
        return "Top Copper"

    if "_b_cu" in name or name.endswith("b_cu"):
        return "Bottom Copper"

    if "_f_mask" in name or name.endswith("f_mask"):
        return "Top Solder Mask"

    if "_b_mask" in name or name.endswith("b_mask"):
        return "Bottom Solder Mask"

    if "_f_silkscreen" in name or name.endswith("f_silkscreen"):
        return "Top Silkscreen"

    if "_b_silkscreen" in name or name.endswith("b_silkscreen"):
        return "Bottom Silkscreen"

    if "_edge_cuts" in name or name.endswith("edge_cuts"):
        return "Board Outline / Mechanical"

    if ext == "drl" or "npth" in name or name.endswith("pth"):
        return "Drill Data"

    if path.suffix.lower() in GERBER_EXTENSIONS:
        return "Other / Unclassified Gerber"

    return "Unsupported / Unknown"

def expand_gerber_uploads(files) -> pd.DataFrame:

    rows, seen = [], set()

    for uploaded in files:

        filename = uploaded.name

        if filename.lower().endswith(".zip"):

            try:

                with zipfile.ZipFile(io.BytesIO(uploaded.getvalue())) as archive:

                    for item in archive.infolist():

                        if item.is_dir() or Path(item.filename).suffix.lower() not in GERBER_EXTENSIONS:

                            continue

                        try:

                            data = archive.read(item)

                        except Exception:

                            continue

                        key = (item.filename, len(data))

                        if key in seen:

                            continue

                        seen.add(key)

                        rows.append({

                            "File Name": Path(item.filename).name, "Archive Path": item.filename,

                            "File Size (KB)": round(len(data) / 1024, 2),

                            "File Type": Path(item.filename).suffix.lower(),

                            "Identified Layer": identify_layer(item.filename), "Source": filename,

                            "Read Status": "Empty file" if not data else "Read successfully",

                        })

            except zipfile.BadZipFile:

                st.error(f"{filename} is not a valid ZIP archive.")

            except Exception as exc:

                st.error(f"Could not read {filename}: {exc}")

        else:

            suffix = Path(filename).suffix.lower()

            if suffix not in GERBER_EXTENSIONS:

                continue

            data = uploaded.getvalue()

            key = (filename, len(data))

            if key in seen:

                continue

            seen.add(key)

            rows.append({

                "File Name": Path(filename).name, "Archive Path": Path(filename).name,

                "File Size (KB)": round(len(data) / 1024, 2), "File Type": suffix,

                "Identified Layer": identify_layer(filename), "Source": filename,

                "Read Status": "Empty file" if not data else "Read successfully",

            })

    return pd.DataFrame(rows, columns=["File Name", "Archive Path", "File Size (KB)", "File Type", "Identified Layer", "Source", "Read Status"])





def make_layer_checklist(inventory: pd.DataFrame) -> pd.DataFrame:

    found_layers = set()

    if not inventory.empty:

        found_layers = set(inventory.loc[inventory["Identified Layer"] != "Unsupported / Unknown", "Identified Layer"].tolist())

    rows = []

    for layer in EXPECTED_LAYERS:

        found = layer in found_layers

        rows.append({"Expected Layer": layer, "Status": "Found by filename" if found else "Not identified", "Result": "Present" if found else "Check uploaded files"})

    return pd.DataFrame(rows)





def render_gerber_page() -> None:

    st.markdown('<div class="section-heading">▤ &nbsp; Gerber Manufacturing-File Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-note">Upload individual manufacturing files or a ZIP archive to review likely PCB layers and file inventory.</div>', unsafe_allow_html=True)

    st.info("This module performs filename-based inventory and layer completeness checks. It does not validate Gerber geometry, electrical connectivity, clearances or manufacturing DRC rules.")

    files = st.file_uploader(

        "Upload Gerber files or a ZIP archive",

        type=["zip", "gbr", "ger", "gerber", "gtl", "gbl", "gts", "gbs", "gto", "gbo", "gtp", "gbp", "gko", "gm1", "gml", "g2", "g3", "drl", "xln", "art", "pho", "cmp", "sol", "stc", "sts", "plc", "pls", "outline"],

        accept_multiple_files=True, key="gerber_uploads",

    )

    if not files:

        st.markdown('<div class="feature-card"><h3>Gerber analysis workflow</h3><p>1. Upload manufacturing files or a ZIP archive.<br>2. Identify likely layers from file names and extensions.<br>3. Review the expected-layer checklist.<br>4. Download inventory and checklist CSV reports.</p></div>', unsafe_allow_html=True)

        return

    inventory = expand_gerber_uploads(files)

    if inventory.empty:

        st.warning("No supported Gerber or drill files were found. Check the file extensions or ZIP contents.")

        return

    checklist = make_layer_checklist(inventory)

    identified = int(inventory["Identified Layer"].isin(EXPECTED_LAYERS.keys()).sum())

    present = int((checklist["Status"] == "Found by filename").sum())

    empty = int((inventory["Read Status"] == "Empty file").sum())

    st.divider()

    st.markdown("### Gerber Inspection Summary")

    cols = st.columns(4)

    cols[0].metric("Supported Files", len(inventory))

    cols[1].metric("Identified Layers", identified)

    cols[2].metric("Expected Layers Found", f"{present}/{len(EXPECTED_LAYERS)}")

    cols[3].metric("Empty Files", empty)

    st.markdown("#### Uploaded File Inventory")

    st.dataframe(inventory, use_container_width=True, hide_index=True)

    st.markdown("#### Expected Layer Checklist")

    st.caption("Layer identification is inferred from file naming conventions. Different CAD/export tools may use other naming conventions.")

    st.dataframe(checklist, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)

    c1.download_button(
        "⬇ Download Gerber Inventory CSV",
        inventory.to_csv(index=False).encode("utf-8"),
        "gerber_inventory.csv",
        "text/csv",
        use_container_width=True,
    )

    c2.download_button(
        "⬇ Download Layer Checklist CSV",
        checklist.to_csv(index=False).encode("utf-8"),
        "gerber_layer_checklist.csv",
        "text/csv",
        use_container_width=True,
    )


def open_pcb_page() -> None:
    st.session_state["page_selection"] = "PCB Fault Detection"
    

def open_gerber_page() -> None:
    st.session_state["page_selection"] = "Gerber Analysis"

def sync_navigation() -> None:
    st.session_state["page_selection"] = st.session_state["sidebar_navigation"]
   
def go_home() -> None:
    st.session_state["page_selection"] = "Home"
    st.session_state["sidebar_navigation"] = "Home"

# --------------------------- Dashboard UI ---------------------------

def navigate_to(page: str) -> None:
    st.session_state["page_selection"] = page


if "page_selection" not in st.session_state:
    st.session_state["page_selection"] = "Home"


with st.sidebar:
    logo_uri = image_as_data_uri(LOGO_PATH)

    if logo_uri:
        st.markdown(
            f'<div class="brand-mini"><img src="{logo_uri}" style="width:58px;height:58px;object-fit:contain;background:#ffffff;border:2px solid #20d1dc;border-radius:10px;padding:3px"><div><div class="brand-name">PCB AI</div><div class="brand-sub">INSPECTION STUDIO</div></div></div><div class="brand-tagline">{TAGLINE}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="brand-mini"><div class="brand-chip">▦</div><div><div class="brand-name">PCB AI</div><div class="brand-sub">INSPECTION STUDIO</div></div></div><div class="brand-tagline">Precision · Detection · Inspection</div>',
            unsafe_allow_html=True,
        )

    
        st.markdown("### Navigation")


        st.markdown("### Navigation")

    pages = [
        "Home",
        "PCB Fault Detection",
        "Gerber Analysis",
        "About",
    ]

    if "page_selection" not in st.session_state:
        st.session_state["page_selection"] = "Home"

    if "sidebar_navigation" not in st.session_state:
        st.session_state["sidebar_navigation"] = st.session_state["page_selection"]

    selected = st.radio(
        "Select a module",
        pages,
        index=pages.index(st.session_state["page_selection"]),
        key="sidebar_navigation",
        label_visibility="collapsed",
        on_change=sync_navigation,
    )
   



# --------------------------- Page Content ---------------------------


# --------------------------- Page Content ---------------------------
page = st.session_state.get("page_selection", "Home")

if page == "Home":
    render_hero()

    st.markdown(
        '<div class="section-heading">⌂ &nbsp; Inspection Dashboard</div>'
        '<div class="section-note">Choose an inspection module to begin.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>PCB Image Fault Detection</h3>
                <p>Detect PCB defects using your trained YOLO model
                and SAHI sliced inference.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
    " Open PCB Fault Detection",
    key="open_pcb",
    type="primary",
    use_container_width=True,
    on_click=open_pcb_page,
)

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Gerber File Analysis</h3>
                <p>Review Gerber files, identify likely layers,
                and export inventory reports.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
    " Open Gerber Analysis",
    key="open_gerber",
    type="primary",
    use_container_width=True,
    on_click=open_gerber_page,
)

    st.markdown(
        """
        <div class="feature-card">
            <h3>System Overview</h3>
            <p><strong>AI-Based Defect Detection:</strong>
            Identifies visible PCB faults using a trained AI model.</p>
            <p><strong>Defect Highlighting:</strong>
            Displays detected defects with bounding boxes.</p>
            <p><strong>Confidence Scores:</strong>
            Shows model confidence for each detection.</p>
            <p><strong>Gerber Analysis:</strong>
            Organizes manufacturing files and checks likely layer coverage.</p>
            <p><strong>Faster Inspection:</strong>
            Helps reduce manual inspection effort.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()

elif page == "PCB Fault Detection":
    st.button(
        "← Back to Home",
        key="pcb_back_home",
        on_click=go_home,
    )

    render_hero()
    render_pcb_page()
    render_footer()

elif page == "Gerber Analysis":
    st.button(
            "← Back to Home",
            key="pcb_back_home",
            on_click=go_home,
        )
    render_hero()
    render_gerber_page()
    render_footer()




elif page == "About":
    render_hero()

    st.markdown(
        "<h2 style='color:#082b50;'>About PCB AI Inspection Studio</h2>",
        unsafe_allow_html=True,
    )

    st.write(
        "PCB AI Inspection Studio is designed to automatically inspect "
        "Printed Circuit Boards (PCBs) and detect visible defects."
    )

    st.divider()

    st.subheader("Key Features")

    with st.container(border=True):
        st.markdown("### AI-Based Defect Detection")
        st.write(
            "Identifies visible PCB faults using a trained AI model."
        )

    with st.container(border=True):
        st.markdown("### Defect Highlighting")
        st.write(
            "Highlights detected defects on the PCB image."
        )

    with st.container(border=True):
        st.markdown("### Confidence Scores")
        st.write(
            "Displays confidence scores for detected defects."
        )

    with st.container(border=True):
        st.markdown("### Gerber Analysis")
        st.write(
            "Helps organize Gerber files and identify likely "
            "manufacturing layers from filenames."
        )

    with st.container(border=True):
        st.markdown("### Faster Inspection")
        st.write(
            "Helps reduce manual inspection effort."
        )

    st.divider()
    st.subheader("Technologies Used")

    st.markdown(
        "- YOLOv8 — AI-based defect detection\n"
        "- SAHI — sliced image inference\n"
        "- Streamlit — web interface\n"
        "- Python, Pillow, NumPy and Pandas"
    )

    render_footer()








