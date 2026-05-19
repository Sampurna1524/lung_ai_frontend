import base64
import html
import io
import os
import random
import time

import cv2
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

st.set_page_config(
    page_title="Lung AI System",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🫁"
)

# =====================================
# LOGIN SESSION CHECK
# =====================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""

# REDIRECT TO LOGIN IF NOT ITS NOT AUTHENTICATED

if not st.session_state["logged_in"]:
    st.switch_page("login.py")

# 🔥 Backend health check
try:
    res = requests.get("http://127.0.0.1:8000/health", timeout=2)
    if res.status_code == 200:
        st.success("🟢 Backend Connected")
    else:
        st.warning("🟡 Backend not responding properly")
except:
    st.error("🔴 Backend not running")
    

# 🔥 Initialize session state (GLOBAL APP STATE)

if "image" not in st.session_state:
    st.session_state["image"] = None

if "metadata" not in st.session_state:
    st.session_state["metadata"] = {}

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

if "confidence" not in st.session_state:
    st.session_state["confidence"] = None

if "xai" not in st.session_state:
    st.session_state["xai"] = None

# 🔥 Initialize slideshow state
if "show_slideshow" not in st.session_state:
    st.session_state["show_slideshow"] = False

if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

if "auto_play" not in st.session_state:
    st.session_state["auto_play"] = True

if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = None

if "inference_done" not in st.session_state:
    st.session_state["inference_done"] = False

if "show_inline_analytics" not in st.session_state:
    st.session_state["show_inline_analytics"] = False

if "inference_error" not in st.session_state:
    st.session_state["inference_error"] = None

# ================================
# PREPROCESSING HELPER FUNCTIONS
# ================================

def to_numpy(img):
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    return np.array(Image.fromarray(img).convert("RGB"))

def to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def resize(img):
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

def clahe(img):
    gray = to_gray(img)
    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_filter.apply(gray)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def median_filter(img):
    return cv2.medianBlur(img, 5)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def canny_edge(img):
    gray = to_gray(img)
    return cv2.Canny(gray, 100, 200)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def gamma_correction(img, gamma=1.5):
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.uint8)
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def to_data_uri(img):
    """Convert image data into a PNG data URI for animated HTML slides."""
    if img is None:
        return ""

    image_array = to_numpy(img).astype(np.uint8)
    pil_img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def build_inference_payload(image, model_name):
    model_map = {
        "EfficientNetV2-S": "efficientnet",
        "Vision Transformer": "vit",
        "ResNet18 BYOL": "byol",
        "SwAV ResNet18": "swav",
        "Ensemble Model": "ensemble"
    }

    if model_name not in model_map:
        raise ValueError("Please choose a valid model.")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im, model_map[model_name].strip().lower()

def request_model_inference(image_bytes, model_slug):
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"model": model_slug}

    res = requests.post(
        "http://127.0.0.1:8000/api/predict",
        files=files,
        data=data,
        timeout=30
    )

    if res.status_code != 200:
        raise RuntimeError("Backend error (non-200 response).")

    output = res.json()

    if "error" in output:
        raise RuntimeError(output["error"])

    prediction = output.get("prediction")
    confidence = output.get("confidence")
    xai = output.get("xai")

    if prediction is None or confidence is None:
        raise RuntimeError("Invalid response from backend.")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "xai": xai if xai and isinstance(xai, dict) and len(xai) > 0 else None,
    }

def run_model_inference(model_name):
    """Run backend inference for the selected model and store the response in session state."""
    if st.session_state.get("image") is None:
        raise ValueError("Please upload an image first.")

    image_bytes, model_slug = build_inference_payload(st.session_state["image"], model_name)
    output = request_model_inference(image_bytes, model_slug)

    st.session_state["prediction"] = output["prediction"]
    st.session_state["confidence"] = output["confidence"]
    st.session_state["model_used"] = model_name
    st.session_state["xai"] = output["xai"]
    st.session_state["inference_done"] = True

    return output["prediction"], output["confidence"]

def resolve_backend_image_path(path):
    if not path:
        return None
    safe_path = path.replace("\\", "/")
    if not os.path.isabs(safe_path):
        safe_path = os.path.join(os.getcwd(), "backend", safe_path)
    return safe_path if os.path.exists(safe_path) else None

def load_xai_image(path):
    resolved = resolve_backend_image_path(path)
    if not resolved:
        return None
    image = cv2.imread(resolved)
    if image is None:
        return None
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_hotspot_summary(pil_img):
    if pil_img is None:
        return "No clear hotspot region was available for this overlay."

    arr = np.array(pil_img.convert("RGB"))
    intensity = arr[:, :, 0].astype(np.float32) + arr[:, :, 1].astype(np.float32) * 0.7
    y, x = np.unravel_index(np.argmax(intensity), intensity.shape)
    height, width = intensity.shape

    vertical = "upper" if y < height / 3 else "lower" if y > (2 * height / 3) else "central"
    horizontal = "left" if x < width / 2 else "right"
    return f"strongest around the {vertical}-{horizontal} lung region"

def get_prediction_context(prediction):
    label = str(prediction or "").lower()

    if "adeno" in label:
        return {
            "daily_life": "This pattern can line up with shortness of breath, fatigue, a persistent cough, and reduced tolerance for walking or stairs when symptoms are active.",
            "extra": "Adenocarcinoma is one of the more common lung cancer subtypes and can arise in the outer parts of the lung, which is why imaging review matters.",
        }
    if "squamous" in label:
        return {
            "daily_life": "People can experience cough, chest discomfort, wheezing, or irritation during routine activity if central airways are involved.",
            "extra": "Squamous cell lung cancer is often discussed as a more centrally located pattern, so airway symptoms can become more noticeable in daily life.",
        }
    if "small" in label:
        return {
            "daily_life": "This kind of pattern can affect energy, breathing comfort, and day-to-day stamina quite quickly if it is truly aggressive.",
            "extra": "Small cell lung cancer is known for fast growth, which is why rapid follow-up is usually emphasized when imaging and clinical signs line up.",
        }
    if "large" in label:
        return {
            "daily_life": "Possible effects include breathlessness, low exercise tolerance, and chest heaviness, depending on where the abnormal tissue sits.",
            "extra": "Large cell lung cancer is a broader category and can appear in different lung regions, so image context and pathology matter a lot.",
        }
    if "benign" in label:
        return {
            "daily_life": "Benign findings often have a lighter day-to-day impact, though symptoms can still come from inflammation, infection, or airway irritation.",
            "extra": "A benign-leaning AI output still deserves clinical review because non-cancer lung findings can look noisy on scans.",
        }
    if "normal" in label or "negative" in label:
        return {
            "daily_life": "A normal-leaning result suggests the scan does not show a strong concerning pattern, though symptoms should still be reviewed with the full clinical picture.",
            "extra": "Normal AI outputs are reassuring but not final diagnosis; imaging, symptoms, and physician review still work together.",
        }

    return {
        "daily_life": "If this pattern reflects a real lung abnormality, daily-life effects could include cough, reduced stamina, chest discomfort, or shortness of breath depending on severity.",
        "extra": "This AI label should be treated as a screening-style suggestion rather than a standalone diagnosis.",
    }

def get_xai_descriptions(hotspot_text):
    return {
        "gradcam": {
            "title": "Grad-CAM Overlay",
            "summary": f"This heatmap shows where the model concentrated its attention, with the AI focus appearing {hotspot_text}.",
            "meaning": "Brighter regions usually mean the model leaned more heavily on those structures when deciding the class.",
        },
        "saliency": {
            "title": "Saliency Map",
            "summary": f"This view highlights the pixels that most strongly changed the model output, again pointing {hotspot_text}.",
            "meaning": "Sharper contrast here suggests the decision is sensitive to local edges, textures, or density changes in that area.",
        },
        "symmetry": {
            "title": "Symmetry Analysis",
            "summary": "This comparison looks for left-right imbalance, which can help surface unusual density or shape differences between lungs.",
            "meaning": "When one side behaves differently from the other, it can be a useful clue for clinicians to inspect that region more closely.",
        },
    }

def build_report_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    metadata = st.session_state.get("metadata", {})
    prediction = st.session_state.get("prediction", "N/A")
    confidence = st.session_state.get("confidence", 0.0)
    model_used = st.session_state.get("model_used", st.session_state.get("selected_model", "N/A"))
    patient_name = metadata.get("name", "N/A")
    context = get_prediction_context(prediction)
    xai = st.session_state.get("xai") or {}

    hotspot_text = "within the highlighted lung region"
    gradcam_img = load_xai_image(xai.get("gradcam")) if "gradcam" in xai else None
    if gradcam_img is not None:
        hotspot_text = get_hotspot_summary(gradcam_img)
    xai_text = get_xai_descriptions(hotspot_text)

    story.append(Paragraph("Lung AI Clinical Summary", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Patient Name:</b> {html.escape(str(patient_name))}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Age:</b> {metadata.get('age', 'N/A')}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Gender:</b> {html.escape(str(metadata.get('gender', 'N/A')))}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Smoking History:</b> {html.escape(str(metadata.get('smoking', 'N/A')))}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Symptoms:</b> {html.escape(str(metadata.get('symptoms', 'N/A')))}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Prediction:</b> {html.escape(str(prediction))}", styles["Heading3"]))
    story.append(Paragraph(f"<b>Confidence:</b> {float(confidence):.2f}%", styles["BodyText"]))
    story.append(Paragraph(f"<b>Model Used:</b> {html.escape(str(model_used))}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("AI-Guided Interpretation", styles["Heading2"]))
    story.append(Paragraph(html.escape(context["daily_life"]), styles["BodyText"]))
    story.append(Paragraph(html.escape(context["extra"]), styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Clinical Reading Support", styles["Heading2"]))
    story.append(Paragraph(
        f"The model's strongest attention pattern appears {html.escape(hotspot_text)}. "
        "This should be treated as a review aid that helps a clinician decide where to inspect the scan more closely.",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        "Potential day-to-day effects can include cough, fatigue, lower activity tolerance, chest discomfort, or breathlessness depending on the true diagnosis and disease burden.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Explainability Note", styles["Heading2"]))
    story.append(Paragraph("These overlays show where the model paid the most attention. They support review but do not replace clinical diagnosis or pathology.", styles["BodyText"]))

    for method in ["gradcam", "saliency", "symmetry"]:
        if method not in xai:
            continue
        pil_img = load_xai_image(xai.get(method))
        if pil_img is None:
            continue

        story.append(Spacer(1, 10))
        story.append(Paragraph(xai_text[method]["title"], styles["Heading3"]))
        story.append(Paragraph(html.escape(xai_text[method]["summary"]), styles["BodyText"]))
        story.append(Paragraph(html.escape(xai_text[method]["meaning"]), styles["BodyText"]))

        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        story.append(Spacer(1, 6))
        story.append(ReportImage(img_buffer, width=220, height=220))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Report Disclaimer", styles["Heading2"]))
    story.append(Paragraph(
        "This PDF is generated from AI outputs and preprocessing artifacts for assistive review only. Final interpretation should come from qualified medical professionals using the full clinical context.",
        styles["BodyText"]
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

@st.dialog("System Info", width="large")
def show_system_info():
    st.markdown("""
    ### How This System Works

    This application walks through a full AI-assisted lung scan review flow:

    1. **Upload and Metadata**  
    The scan image and patient details are captured first so the analysis has the clinical context it needs.

    2. **Preprocessing Pipeline**  
    The slideshow previews normalization, enhancement, filtering, and structural feature steps that help the model work from a cleaner image.

    3. **Model Inference**  
    The selected deep learning model evaluates the scan and returns a predicted label, confidence score, and explainability outputs when available.

    4. **Explainable AI Review**  
    Grad-CAM, saliency, and symmetry overlays help show *why* the model focused on a region and where review attention may be warranted.

    5. **Inline Analytics and PDF Report**  
    Results can be reviewed directly on the home screen and exported as a richer report with patient details, interpretation notes, and XAI figures.

    ### What Each AI View Means

    - **Grad-CAM:** Shows the regions the model considered most influential.
    - **Saliency Map:** Highlights pixels that most strongly affect the output.
    - **Symmetry Analysis:** Helps surface left-right differences that may deserve closer review.

    ### Important Note

    This system is designed for **assistive review**, not final diagnosis.  
    The overlays show model attention, not guaranteed tumor boundaries, and all outputs should be interpreted by qualified clinicians.
    """)

def generate_preprocessing_steps():
    """Generate all preprocessing steps"""
    if st.session_state.get("image") is None:
        return []
    
    original = to_numpy(st.session_state["image"])
    
    steps = [
        ("Original Scan", original, "Your uploaded lung scan image"),
    ]
    
    # Basic preprocessing
    resized = resize(original)
    steps.append(("Resized (224x224)", resized, "Image resized to standard size for model processing"))
    
    # Augmentation
    flipped = cv2.flip(resized, 1)
    steps.append(("Horizontal Flip", flipped, "Data augmentation: horizontal flip"))
    
    # Self-Supervised crops
    crop1 = resized[0:200, 0:200]
    steps.append(("Crop 1", crop1, "Self-supervised learning: first crop region"))
    
    crop2 = resized[20:220, 20:220]
    steps.append(("Crop 2", crop2, "Self-supervised learning: second crop region"))
    
    # Medical processing
    clahe_img = clahe(resized)
    steps.append(("CLAHE Enhancement", clahe_img, "Contrast enhancement for medical imaging"))
    
    # Enhancement filters
    gaussian = gaussian_blur(clahe_img)
    steps.append(("Gaussian Blur", gaussian, "Noise reduction with Gaussian blur"))
    
    median = median_filter(clahe_img)
    steps.append(("Median Filter", median, "Noise reduction with median filter"))
    
    bilateral = bilateral_filter(resized)
    steps.append(("Bilateral Filter", bilateral, "Edge-preserving bilateral filter"))
    
    gamma = gamma_correction(resized)
    steps.append(("Gamma Correction", gamma, "Brightness adjustment and enhancement"))
    
    sharpened = sharpen(resized)
    steps.append(("Sharpening", sharpened, "Edge enhancement with sharpening kernel"))
    
    # Structural features
    edges = canny_edge(clahe_img)
    steps.append(("Canny Edges", edges, "Edge detection: identifies structural boundaries"))
    
    return steps

st.markdown("""
<style>

/* SIDEBAR BASE */

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#050B1E,#020617);
    border-right:1px solid rgba(0,255,255,0.15);
}

/* SIDEBAR TITLE */

.sidebar-title{
    font-size:22px;
    font-weight:700;
    color:#00eaff;
    text-align:center;
    padding-bottom:10px;
}

/* SIDEBAR CARD */

.sidebar-card{
    background:rgba(255,255,255,0.04);
    border-radius:14px;
    padding:12px;
    margin-bottom:10px;
    border:1px solid rgba(0,255,255,0.15);
    transition:0.3s;
}

.sidebar-card:hover{
    transform:translateX(4px);
    border:1px solid #00eaff;
    box-shadow:0 0 10px rgba(0,234,255,0.3);
}

/* NAV BUTTONS */

.sidebar-button button{
    width:100%;
    border-radius:12px;
    border:none;
    background:rgba(0,234,255,0.08);
    color:white;
    padding:10px;
    margin-top:6px;
    transition:0.3s;
}

.sidebar-button button:hover{
    background:#00eaff;
    color:black;
}

/* DIVIDER */

.sidebar-divider{
    border-top:1px solid rgba(255,255,255,0.1);
    margin:12px 0;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# STREAMLIT BACKGROUND FIX
# -----------------------------

st.markdown("""
<style>

[data-testid="stSidebar"]{
display:none !important;
}

[data-testid="collapsedControl"]{
display:none !important;
}

section[data-testid="stSidebar"]{
display:none !important;
max-width:0px !important;
min-width:0px !important;
width:0px !important;
}

button[kind="header"]{
display:none !important;
}

[data-testid="stToolbar"]{
display:none !important;
}
            
/* FLOATING INFO BUTTON */

.st-key-floating_info_button div[data-testid="stButton"]{
    position: fixed !important;
    bottom: 20px !important;
    right: 20px !important;
    z-index: 9999 !important;
    width: auto !important;
}

/* BUTTON STYLE */

.st-key-floating_info_button button{
    width: 80px !important;
    height: 40px !important;
    min-height: 40px !important;

    border-radius: 12px !important;

    background: linear-gradient(135deg,#00d4ff,#6c63ff) !important;
    color: white !important;

    font-size: 14px !important;
    font-weight: 600 !important;

    border: none !important;

    box-shadow: 0 0 14px rgba(0,224,255,0.45) !important;

    transition: 0.3s ease !important;

    animation: infoBlink 2s infinite;
}

/* HOVER EFFECT */

.st-key-floating_info_button button:hover{
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0,224,255,0.7) !important;
}

/* BLINK */

@keyframes infoBlink{
    0%,100%{
        transform: scale(1);
        opacity: 1;
    }
    50%{
        transform: scale(1.08);
        opacity: 0.8;
    }
}

html, body {
background:#020617 !important;
}

.stApp{
background:transparent !important;
}

[data-testid="stAppViewContainer"]{
background:transparent !important;
}

[data-testid="stHeader"]{
background:transparent !important;
}

iframe{
position:fixed !important;
top:0;
left:0;
width:100vw !important;
height:100vh !important;
border:none !important;
z-index:-1 !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# BACKGROUND
# -----------------------------

background = """
<!DOCTYPE html>
<html>
<head>

<style>

html,body{
margin:0;
overflow:hidden;
background:#020617;
}

#particles-js{
position:fixed;
width:100%;
height:100%;
top:0;
left:0;
z-index:-2;
}

#hologram{
position:fixed;
top:50%;
left:50%;
transform:translate(-50%,-50%);
width:900px;
height:900px;
opacity:0.35;
pointer-events:none;
}

</style>

</head>

<body>

<div id="particles-js"></div>
<div id="hologram"></div>

<script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/loaders/GLTFLoader.js"></script>

<script>

/* PARTICLES */

particlesJS("particles-js", {
"particles":{
"number":{"value":80},
"color":{"value":"#00E0FF"},
"shape":{"type":"circle"},
"opacity":{"value":0.5},
"size":{"value":3},
"line_linked":{
"enable":true,
"distance":150,
"color":"#00E0FF",
"opacity":0.3,
"width":1
},
"move":{"enable":true,"speed":2}
}
});


/* THREEJS */

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45,1,0.1,1000);

const renderer = new THREE.WebGLRenderer({
alpha:true,
antialias:true
});

renderer.setSize(900,900);

document.getElementById("hologram").appendChild(renderer.domElement);


/* LIGHTS */

const light = new THREE.PointLight(0x00ffff,1.5);
light.position.set(10,10,10);
scene.add(light);

scene.add(new THREE.AmbientLight(0x404040));


/* LOAD MODEL */

const loader = new THREE.GLTFLoader();

let torso;

loader.load(
"https://modelviewer.dev/shared-assets/models/Astronaut.glb",
function(gltf){

torso = gltf.scene;

/* SCALE MODEL */

torso.scale.set(5,5,5);


/* CENTER MODEL */

const box = new THREE.Box3().setFromObject(torso);
const center = box.getCenter(new THREE.Vector3());

torso.position.x += (torso.position.x - center.x);
torso.position.y += (torso.position.y - center.y);
torso.position.z += (torso.position.z - center.z);


/* MOVE BODY DOWN */

torso.position.y -= 1.5;


/* HOLOGRAM MATERIAL */

torso.traverse(function(child){

if(child.isMesh){

child.material = new THREE.MeshBasicMaterial({
color:0x00ffff,
wireframe:true,
transparent:true,
opacity:0.35
});

}

});

scene.add(torso);

}
);


/* CAMERA */

camera.position.set(0,0,8);


/* CURSOR INTERACTION */

let mouseX = 0;
let mouseY = 0;

window.parent.addEventListener("mousemove",(event)=>{

mouseX = (event.clientX/window.innerWidth)*2-1;
mouseY = -(event.clientY/window.innerHeight)*2+1;

});


/* ANIMATION */

function animate(){

requestAnimationFrame(animate);

if(torso){

torso.rotation.x = mouseY * 0.3;
torso.rotation.y = mouseX * 0.3;

}

renderer.render(scene,camera);

}

animate();

</script>

</body>
</html>
"""

components.html(background, height=0)


# -----------------------------
# LOAD CARD CSS
# -----------------------------

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------
# UI
# -----------------------------

top_col1, top_col2 = st.columns([6,1])

with top_col1:

    st.title("🫁 Multimodal Lung Cancer AI System")

    st.markdown(
        f"""
        ### Hello, {st.session_state['username']}! 👋

        AI-Driven MRI Analysis using Deep Learning and Explainable AI
        """
    )

with top_col2:

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state["logged_in"] = False
        st.session_state["username"] = ""

        st.switch_page("login.py")

st.markdown(
"""
This platform demonstrates an **automated lung cancer detection pipeline**
that integrates advanced **image preprocessing, deep learning models, and
explainable AI techniques** to assist clinicians in medical decision-making.
"""
)

st.divider()

# ---------------------------------
# UPLOAD + METADATA + MODEL SELECTION
# ---------------------------------

st.subheader("📋 Setup Your Analysis")

col1, col2, col3 = st.columns(3)

# UPLOAD IMAGE
with col1:
    st.markdown("### 📂 Upload Image")
    uploaded = st.file_uploader(
        "Upload Lung Scan",
        type=["png", "jpg", "jpeg"],
        help="Upload a lung MRI or CT scan image file here."
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Scan Preview", use_container_width=True)
        st.session_state["image"] = img
        st.success("✓ Image uploaded")
    else:
        if st.session_state.get("image") is not None:
            st.success("✓ Image on file")
        else:
            st.info("No image uploaded yet")

# PATIENT METADATA
with col2:
    st.markdown("### 👤 Patient Metadata")
    patient_name = st.text_input("Patient Name", value=st.session_state.get("metadata", {}).get("name", ""))
    age = st.number_input("Age", 0, 120, value=st.session_state.get("metadata", {}).get("age", 35))
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
        index=["Male", "Female", "Other"].index(st.session_state.get("metadata", {}).get("gender", "Male"))
    )
    smoking = st.selectbox(
        "Smoking History",
        ["Yes", "No"],
        index=["Yes", "No"].index(st.session_state.get("metadata", {}).get("smoking", "No"))
    )
    symptoms = st.text_area(
        "Clinical Symptoms",
        value=st.session_state.get("metadata", {}).get("symptoms", ""),
        height=100
    )

    st.session_state["metadata"] = {
        "name": patient_name,
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "symptoms": symptoms
    }

    if patient_name and symptoms:
        st.success("✓ Metadata complete")
    else:
        st.warning("Add patient name and symptoms to proceed")

# MODEL SELECTION
with col3:
    st.markdown("### 🤖 Choose Model")
    model_options = [
        "EfficientNetV2-S",
        "Vision Transformer",
        "ResNet18 BYOL",
        "SwAV ResNet18",
        "Ensemble Model"
    ]
    default_index = 0
    if st.session_state.get("selected_model") in model_options:
        default_index = model_options.index(st.session_state["selected_model"])

    selected_model = st.selectbox(
        "Select AI Model",
        model_options,
        index=default_index
    )
    st.session_state["selected_model"] = selected_model
    
    st.info(f"**{selected_model}** selected for inference")
    
    st.markdown("**Models:**")
    st.caption("• EfficientNetV2-S - Fast & accurate")
    st.caption("• ViT - Transformer-based")
    st.caption("• BYOL - Self-supervised")
    st.caption("• SwAV - Contrastive learning")
    st.caption("• Ensemble - Combined models")

st.markdown("---")

proceed_disabled = not (
    st.session_state.get("image") is not None and
    bool(st.session_state.get("metadata", {}).get("name")) and
    bool(st.session_state.get("metadata", {}).get("symptoms")) and
    st.session_state.get("selected_model") is not None
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Proceed to Preprocessing", disabled=proceed_disabled, use_container_width=True):
        st.session_state["show_slideshow"] = True
        st.session_state["current_step"] = 0
        st.session_state["auto_play"] = True
        st.session_state["prediction"] = None
        st.session_state["confidence"] = None
        st.session_state["xai"] = None
        st.session_state["inference_done"] = False
        st.session_state["inference_error"] = None
        st.session_state["show_inline_analytics"] = False
        st.rerun()

with col2:
    if st.session_state.get("image"):
        st.info(f"Ready: Image + Metadata + {st.session_state.get('selected_model', 'Model')}")

# ================================
# PREPROCESSING SLIDESHOW
# ================================

if st.session_state.get("show_slideshow"):
    st.divider()
    st.markdown("## 🎬 Preprocessing Pipeline Slideshow")
    
    st.markdown("""
    <style>
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes revealPanel {
        from {
            opacity: 0;
            transform: scale(0.985);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    @keyframes imageFloat {
        0% {
            transform: scale(1.02) translateY(0);
        }
        50% {
            transform: scale(1.05) translateY(-6px);
        }
        100% {
            transform: scale(1.02) translateY(0);
        }
    }

    @keyframes countdown {
        from {
            width: 100%;
        }
        to {
            width: 0%;
        }
    }

    .slide-shell {
        animation: revealPanel 0.6s ease-out;
    }

    .slide-stage {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        border: 1px solid rgba(0, 234, 255, 0.22);
        background:
            radial-gradient(circle at top right, rgba(108, 99, 255, 0.18), transparent 42%),
            linear-gradient(180deg, rgba(9, 18, 38, 0.96), rgba(3, 10, 24, 0.96));
        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
    }

    .slide-stage::after {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(180deg, rgba(255,255,255,0.08), transparent 24%, transparent 72%, rgba(0,0,0,0.14));
    }

    .slide-visual {
        height: 380px;
        padding: 18px 18px 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .slide-visual img {
        display: block;
        width: 100%;
        max-width: 540px;
        max-height: 100%;
        margin: 0 auto;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        animation: imageFloat 3.2s ease-in-out infinite;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
        object-fit: contain;
    }

    .slide-meta {
        padding: 18px 22px 22px;
        animation: fadeUp 0.7s ease-out 0.1s both;
    }

    .slide-kicker {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }

    .slide-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(0, 234, 255, 0.12);
        border: 1px solid rgba(0, 234, 255, 0.2);
        color: #8eeaff;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0;
    }

    .slide-title {
        margin: 0;
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 700;
    }

    .slide-desc {
        margin: 10px 0 0;
        color: #d4eaff;
        line-height: 1.55;
        font-size: 0.98rem;
    }

    .slide-progress {
        position: relative;
        margin-top: 18px;
        height: 6px;
        width: 100%;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(255,255,255,0.08);
    }

    .slide-progress-bar {
        position: absolute;
        inset: 0 auto 0 0;
        border-radius: inherit;
        background: linear-gradient(90deg, #00eaff, #6c63ff);
        width: 100%;
    }

    .slide-progress-bar.is-playing {
        animation: countdown 3s linear forwards;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Generate preprocessing steps
    if st.session_state["processed_images"] is None:
        st.session_state["processed_images"] = generate_preprocessing_steps()

    steps = st.session_state["processed_images"]
    
    if steps:
        current_idx = st.session_state.get("current_step", 0)
        total_steps = len(steps)
        
        # Slideshow controls
        col_prev, col_info, col_next, col_auto = st.columns([1, 2, 1, 1.5])
        
        with col_prev:
            if st.button("⬅ Previous", use_container_width=True):
                st.session_state["current_step"] = max(0, current_idx - 1)
                st.session_state["auto_play"] = False
                st.rerun()
        
        with col_info:
            st.metric("Step", f"{current_idx + 1} / {total_steps}")
        
        with col_next:
            if st.button("Next ➡", use_container_width=True):
                st.session_state["current_step"] = min(total_steps - 1, current_idx + 1)
                st.session_state["auto_play"] = False
                st.rerun()
        
        with col_auto:
            auto_mode = st.checkbox("Auto Play", value=st.session_state.get("auto_play", True))
            st.session_state["auto_play"] = auto_mode

        st.markdown("---")
        
        # Display current step
        step_title, step_image, step_desc = steps[current_idx]
        slide_uri = to_data_uri(step_image)
        progress_class = "slide-progress-bar is-playing" if st.session_state.get("auto_play") and current_idx < total_steps - 1 else "slide-progress-bar"
        slide_html = f"""
        <div class="slide-shell">
            <div class="slide-stage">
                <div class="slide-visual">
                    <img src="{slide_uri}" alt="{html.escape(step_title)}" />
                </div>
                <div class="slide-meta">
                    <div class="slide-kicker">
                        <span class="slide-pill">Step {current_idx + 1} of {total_steps}</span>
                        <span class="slide-pill">{"Auto Play On" if st.session_state.get("auto_play") else "Manual Mode"}</span>
                    </div>
                    <h3 class="slide-title">{html.escape(step_title)}</h3>
                    <p class="slide-desc">{html.escape(step_desc)}</p>
                    <div class="slide-progress">
                        <div class="{progress_class}"></div>
                    </div>
                </div>
            </div>
        </div>
        """

        col_left, col_center, col_right = st.columns([0.55, 2.3, 0.55])
        with col_center:
            st.markdown(slide_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Auto-play slideshow
        if st.session_state.get("auto_play") and current_idx < total_steps - 1:
            time.sleep(3)
            st.session_state["current_step"] = current_idx + 1
            st.rerun()

slideshow_steps = (
    st.session_state["processed_images"]
    if st.session_state.get("show_slideshow")
    else []
)
slideshow_complete = bool(slideshow_steps) and st.session_state.get("current_step", 0) >= len(slideshow_steps) - 1

if st.session_state.get("show_slideshow"):
    st.markdown("## Model Inference Results")

    if st.session_state.get("inference_error"):
        st.error(f"Inference failed: {st.session_state.get('inference_error')}")

    # SHOW BUTTON ONLY BEFORE INFERENCE
    if not st.session_state.get("inference_done"):

        button_placeholder = st.empty()
        loading_placeholder = st.empty()
        success_placeholder = st.empty()

        if button_placeholder.button(
            "Run Model Inference",
            use_container_width=True
        ):

            # HIDE BUTTON IMMEDIATELY
            button_placeholder.empty()

            status_message = random.choice([
                "Reviewing your vitals and scan history...",
                "Analysing the scan for suspicious lung patterns...",
                "Cross-checking features with the selected model...",
                "Preparing a clearer readout for the next step..."
            ])

            try:

                with loading_placeholder.container():

                    with st.spinner("Running model inference..."):

                        st.caption(status_message)

                        run_model_inference(
                            st.session_state.get("selected_model")
                        )

                # REMOVE LOADING MESSAGE
                loading_placeholder.empty()

                # SHOW SUCCESS MESSAGE
                success_placeholder.success(
                    "Inference complete."
                )

                # KEEP FOR 1 SECOND
                time.sleep(1)

                # REMOVE SUCCESS MESSAGE
                success_placeholder.empty()

                # REFRESH TO SHOW RESULTS
                st.rerun()

            except requests.exceptions.Timeout:

                loading_placeholder.empty()

                st.session_state["inference_error"] = (
                    "The inference request timed out. Please try again."
                )

                st.error(st.session_state["inference_error"])

            except Exception as exc:

                loading_placeholder.empty()

                st.session_state["inference_error"] = (
                    f"Inference failed: {exc}"
                )

                st.error(st.session_state["inference_error"])


    if st.session_state.get("inference_done"):
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        prediction = st.session_state.get("prediction")
        confidence = st.session_state.get("confidence")
        model_used = st.session_state.get("model_used", st.session_state.get("selected_model", "Selected Model"))
        prediction_context = get_prediction_context(prediction)

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            st.metric("Prediction", prediction)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.caption(f"Model used: {model_used}")

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Confidence Score"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                margin={"l": 20, "r": 20, "t": 50, "b": 20}
            )
            st.plotly_chart(fig, use_container_width=True)

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Analytics Dashboard", use_container_width=True):
                st.session_state["show_inline_analytics"] = not st.session_state.get("show_inline_analytics", False)
        with action_col2:
            st.download_button(
                "Generate Report",
                data=build_report_pdf(),
                file_name="lung_ai_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        if st.session_state.get("show_inline_analytics"):
            st.markdown("### Inline Analytics Dashboard")

            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.metric("Patient", st.session_state.get("metadata", {}).get("name", "N/A"))
            info_col2.metric("Prediction", prediction)
            info_col3.metric("Confidence", f"{confidence:.2f}%")

            st.info(prediction_context["daily_life"])
            st.caption(prediction_context["extra"])

            trend = np.clip(np.linspace(max(confidence - 12, 55), confidence, 8) + np.random.uniform(-2, 2, 8), 0, 100)
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=[f"Review {idx + 1}" for idx in range(len(trend))],
                y=trend,
                mode="lines+markers",
                line={"color": "#00eaff", "width": 3},
                marker={"size": 8}
            ))
            trend_fig.update_layout(
                title="Confidence Trend Snapshot",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
                xaxis_title="Checkpoints",
                yaxis_title="Confidence (%)"
            )
            st.plotly_chart(trend_fig, use_container_width=True)

            st.markdown("### Explainable AI Review")
            xai = st.session_state.get("xai") or {}
            xai_order = ["gradcam", "saliency", "symmetry"]
            xai_cards = []
            primary_hotspot = "within the highlighted lung region"

            for method in xai_order:
                if method in xai:
                    pil_img = load_xai_image(xai.get(method))
                    if pil_img is not None:
                        if method == "gradcam":
                            primary_hotspot = get_hotspot_summary(pil_img)
                        xai_cards.append((method, pil_img))

            xai_text = get_xai_descriptions(primary_hotspot)

            if xai_cards:
                xai_cols = st.columns(len(xai_cards))
                for idx, (method, pil_img) in enumerate(xai_cards):
                    details = xai_text[method]
                    with xai_cols[idx]:
                        st.image(pil_img, use_container_width=True, caption=details["title"])
                        st.markdown(f"**What it shows:** {details['summary']}")
                        st.caption(details["meaning"])
            else:
                st.warning("No explainable AI overlays were available from the current inference run.")

            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.markdown("### Region of Concern")
                st.write(
                    f"The model's strongest visual attention appears {primary_hotspot}. "
                    "That is the area a clinician would usually review first alongside the original scan."
                )
                st.caption("This is an AI attention summary, not a confirmed tumor boundary.")

            with summary_col2:
                st.markdown("### Day-to-Day Impact")
                st.write(prediction_context["daily_life"])
                st.caption("Symptoms and severity can vary a lot by patient and should be clinically confirmed.")

st.markdown('<div class="floating-info-trigger"></div>', unsafe_allow_html=True)
if st.button("Info", key="floating_info_button"):
    show_system_info()
