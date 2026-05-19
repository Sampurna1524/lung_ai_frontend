from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.preprocessing_service import preprocess_image
from services.model_service import predict
from services.xai_service import generate_xai

import numpy as np
import cv2

router = APIRouter()

# 🔥 toggle debug logs
DEBUG = False


@router.post("/predict")
async def run_inference(
    file: UploadFile = File(...),
    model: str = Form(...)
):
    # 🔥 normalize model name (IMPORTANT FIX)
    model = model.strip().lower()

    print("📥 Model received from frontend:", model)

    try:
        # =========================
        # 🔥 READ IMAGE
        # =========================
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file received")

        # =========================
        # 🔥 CREATE ORIGINAL IMAGE (FOR XAI OVERLAY)
        # =========================
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # resize to match model input
        original_img = cv2.resize(original_img, (224, 224))

        # =========================
        # 🔥 PREPROCESS (FOR MODEL)
        # =========================
        processed = preprocess_image(image_bytes)

        if processed is None:
            raise HTTPException(status_code=500, detail="Preprocessing failed")

        # =========================
        # 🔥 MODEL INFERENCE
        # =========================
        result = predict(model, processed, file.filename)

        if not isinstance(result, dict) or "prediction" not in result:
            raise HTTPException(status_code=500, detail="Model inference failed")

        # =========================
        # 🔥 XAI GENERATION (FIXED)
        # =========================
        xai = None

        try:
            # 🔥 PASS ORIGINAL IMAGE (CRITICAL FIX)
            xai = generate_xai(model, processed, original_img)

            if isinstance(xai, dict) and "error" in xai:
                if DEBUG:
                    print("XAI Error:", xai["error"])
                xai = None

        except Exception as xai_err:
            if DEBUG:
                print("XAI Exception:", str(xai_err))
            xai = None

        # =========================
        # 🔥 FINAL RESPONSE
        # =========================
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "xai": xai
        }

    # =========================
    # 🔴 ERROR HANDLING
    # =========================
    except HTTPException as http_err:
        return {"error": http_err.detail}

    except Exception as e:
        if DEBUG:
            return {"error": str(e)}
        return {"error": "Internal server error"}