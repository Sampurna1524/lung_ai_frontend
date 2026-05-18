import torch
import os
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16

MODELS = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")


def load_models():
    global MODELS

    print("🔄 Loading models...")

    # =========================
    # EfficientNet
    # =========================
    try:
        effnet = models.efficientnet_v2_s()
        effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 2)

        effnet.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "effnet_BEST (2).pth"),
            map_location=DEVICE
        ))

        effnet.to(DEVICE).eval()
        MODELS["efficientnet"] = effnet
        print("✅ Loaded: efficientnet")

    except Exception as e:
        print("❌ EfficientNet failed:", e)

    # =========================
    # Vision Transformer (FIXED)
    # =========================
    try:
        vit = vit_b_16(weights=None)
        vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)

        vit.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "best_vit.pth"),
            map_location=DEVICE
        ), strict=False)  # 🔥 FORCE LOAD

        vit.to(DEVICE).eval()
        MODELS["vit"] = vit
        print("✅ Loaded: vit")

    except Exception as e:
        print("❌ ViT failed:", e)

    # =========================
    # MobileNet
    # =========================
    try:
        mobilenet = models.mobilenet_v2()
        mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 2)

        mobilenet.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "mobilenetv2_best.pth"),
            map_location=DEVICE
        ))

        mobilenet.to(DEVICE).eval()
        MODELS["mobilenet"] = mobilenet
        print("✅ Loaded: mobilenet")

    except Exception as e:
        print("❌ MobileNet failed:", e)

    # =========================
    # BYOL
    # =========================
    try:
        byol = models.resnet18()
        byol.fc = nn.Linear(byol.fc.in_features, 2)

        byol.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "best_byol_model.pth"),
            map_location=DEVICE
        ), strict=False)

        byol.to(DEVICE).eval()
        MODELS["byol"] = byol
        print("✅ Loaded: byol")

    except Exception as e:
        print("❌ BYOL failed:", e)

    # =========================
    # SWAV
    # =========================
    try:
        swav = models.resnet18()
        swav.fc = nn.Linear(swav.fc.in_features, 2)

        swav.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "best_swav_model.pth"),
            map_location=DEVICE
        ), strict=False)

        swav.to(DEVICE).eval()
        MODELS["swav"] = swav
        print("✅ Loaded: swav")

    except Exception as e:
        print("❌ SWAV failed:", e)

    # =========================
    # Ensemble
    # =========================
    try:

        ensemble = models.efficientnet_v2_s()

        ensemble.classifier[1] = nn.Linear(
            ensemble.classifier[1].in_features,
            2
        )

        ensemble.load_state_dict(
            torch.load(
                os.path.join(MODEL_DIR, "ensemble.pth"),
                map_location=DEVICE
            )
        )

        ensemble.to(DEVICE).eval()

        MODELS["ensemble"] = ensemble

        print("✅ Loaded: ensemble")

    except Exception as e:

        print("❌ Ensemble failed:", e)

    print("📦 FINAL LOADED MODELS:", list(MODELS.keys()))
    print("🚀 Model loading complete")


def get_model(name):
    if name:
        name = name.strip().lower()

    print("🔍 Requested model:", name)

    model = MODELS.get(name)

    if model is None:
        print("⚠️ Model not found:", name)
        print("Available:", list(MODELS.keys()))

    return model