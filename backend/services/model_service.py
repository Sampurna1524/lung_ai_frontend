import torch
import numpy as np
from models.loader import get_model

# 🔥 device support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model_name, image_tensor):

    try:
        # 🔥 get model
        model = get_model(model_name)

        if model is None:
            return {"error": "Model not found"}

        model.to(DEVICE)

        # 🔥 ensure correct tensor format
        tensor = torch.tensor(image_tensor, dtype=torch.float32)

        # shape check: (C, H, W) → (1, C, H, W)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            output = model(tensor)

        # 🔥 ensure output is tensor
        if not isinstance(output, torch.Tensor):
            return {"error": "Model output is not a tensor"}

        # 🔥 handle different output types
        if len(output.shape) == 2 and output.shape[1] == 1:
            # binary (sigmoid)
            prob = torch.sigmoid(output)
            confidence = float(prob.item() * 100)
            prediction = 1 if prob.item() > 0.5 else 0

        elif len(output.shape) == 2:
            # multi-class (softmax)
            prob = torch.softmax(output, dim=1)
            confidence = float(prob.max() * 100)
            prediction = int(prob.argmax())

        else:
            return {"error": f"Unexpected output shape: {output.shape}"}

        label = "Malignant" if prediction == 1 else "Benign"

        return {
            "prediction": label,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }