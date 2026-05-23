import torch
import os
import torch.nn as nn
from torchvision import models

# =====================================
# MODEL STORAGE
# =====================================

MODELS = {}

# =====================================
# FORCE CPU FOR RENDER DEPLOYMENT
# =====================================

DEVICE = "cpu"

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

# =====================================
# LOAD MODELS
# =====================================

def load_models():

    global MODELS

    print("🔄 Loading models...")
    print(f"🖥 Using device: {DEVICE}")

    # =====================================
    # EFFICIENTNET
    # =====================================

    try:

        effnet = models.efficientnet_v2_s(
            weights=None
        )

        effnet.classifier[1] = nn.Linear(
            effnet.classifier[1].in_features,
            2
        )

        effnet.load_state_dict(
            torch.load(
                os.path.join(
                    MODEL_DIR,
                    "effnet_BEST (2).pth"
                ),
                map_location=DEVICE
            )
        )

        effnet.to(DEVICE)
        effnet.eval()

        MODELS["efficientnet"] = effnet

        print("✅ Loaded: efficientnet")

    except Exception as e:

        print("❌ EfficientNet failed:", e)

    # =====================================
    # MOBILENETV2
    # =====================================

    try:

        mobilenet = models.mobilenet_v2(
            weights=None
        )

        mobilenet.classifier[1] = nn.Linear(
            mobilenet.classifier[1].in_features,
            2
        )

        mobilenet.load_state_dict(
            torch.load(
                os.path.join(
                    MODEL_DIR,
                    "mobilenetv2_best.pth"
                ),
                map_location=DEVICE
            )
        )

        mobilenet.to(DEVICE)
        mobilenet.eval()

        MODELS["mobilenet"] = mobilenet

        print("✅ Loaded: mobilenet")

    except Exception as e:

        print("❌ MobileNet failed:", e)

    # =====================================
    # FINAL STATUS
    # =====================================

    print(
        "📦 FINAL LOADED MODELS:",
        list(MODELS.keys())
    )

    print("🚀 Model loading complete")


# =====================================
# GET MODEL
# =====================================

def get_model(name):

    if name:

        name = name.strip().lower()

    print("🔍 Requested model:", name)

    model = MODELS.get(name)

    if model is None:

        print("⚠️ Model not found:", name)

        print(
            "Available:",
            list(MODELS.keys())
        )

    return model
