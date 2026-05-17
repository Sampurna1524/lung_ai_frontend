import cv2
import numpy as np


def preprocess_image(image_bytes):
    try:
        # 🔥 convert bytes → numpy array
        file_bytes = np.frombuffer(image_bytes, np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image file")

        # 🔥 resize (match model input size)
        img = cv2.resize(img, (224, 224))

        # 🔥 CLAHE (contrast enhancement)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # 🔥 convert BGR → RGB (VERY IMPORTANT)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 🔥 normalize (0–1)
        img = img / 255.0

        # 🔥 OPTIONAL: standard normalization (if used in training)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img = (img - mean) / std

        # 🔥 HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        return img.astype("float32")

    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")