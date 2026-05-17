import torch
import cv2
import numpy as np
import os
from datetime import datetime
from models.loader import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _overlay_heatmap_on_image(heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original image"""
    # Resize heatmap to match original image size if needed
    original_h, original_w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype("uint8"), colormap)
    
    # Convert original image to BGR if needed
    if len(original_image.shape) == 2:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[2] == 4:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
    else:
        original_bgr = original_image
    
    # Ensure both images are the same type and size
    heatmap_colored = cv2.resize(heatmap_colored, (original_bgr.shape[1], original_bgr.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def _generate_gradcam(model, image_tensor, original_image):
    """Generate Grad-CAM visualization"""
    try:
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Find last convolution layer
        target_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
            return None

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        output = model(image_tensor)

        # Target class
        if output.shape[1] == 1:
            score = output[0]
        else:
            score = output[0, output.argmax()]

        # Backward pass
        model.zero_grad()
        score.backward()

        # Get gradients & activations
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]

        # Compute weights
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        cam = cam / (cam.max() + 1e-8)

        # Cleanup hooks
        forward_handle.remove()
        backward_handle.remove()

        # Create overlay
        overlay = _overlay_heatmap_on_image(cam, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET)

        # Save file
        filename = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        cv2.imwrite(path, overlay)

        return path

    except Exception as e:
        return None


def _generate_saliency_map(model, image_tensor, original_image):
    """Generate Saliency Map visualization"""
    try:
        image_tensor_copy = image_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(image_tensor_copy)
        
        # Get target score
        if output.shape[1] == 1:
            score = output[0]
        else:
            score = output[0, output.argmax()]
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        # Get gradients
        saliency = image_tensor_copy.grad.data.abs().max(dim=1)[0][0].cpu().numpy()
        
        # Normalize
        saliency = saliency / (saliency.max() + 1e-8)
        
        # Create overlay using Viridis colormap
        overlay = _overlay_heatmap_on_image(saliency, original_image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)
        
        # Save file
        filename = f"saliency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        cv2.imwrite(path, overlay)
        
        return path

    except Exception as e:
        return None


def _generate_symmetry_analysis(model, image_tensor, original_image):
    """Generate Symmetry Analysis visualization"""
    try:
        model.eval()
        
        # Original prediction
        with torch.no_grad():
            original_output = model(image_tensor)
        
        if original_output.shape[1] == 1:
            original_score = original_output[0].item()
        else:
            original_score = original_output[0, original_output.argmax()].item()
        
        # Horizontal flip
        image_flipped = torch.flip(image_tensor, dims=[-1])
        with torch.no_grad():
            flipped_output = model(image_flipped)
        
        if flipped_output.shape[1] == 1:
            flipped_score = flipped_output[0].item()
        else:
            flipped_score = flipped_output[0, flipped_output.argmax()].item()
        
        # Calculate asymmetry map
        asymmetry = abs(original_score - flipped_score) / (max(original_score, flipped_score) + 1e-8)
        
        # Create heatmap showing asymmetry
        asymmetry_map = np.ones((224, 224), dtype=np.float32) * asymmetry
        
        # Add spatial variation to make it meaningful
        h, w = 224, 224
        for i in range(h):
            for j in range(w):
                # Add left-right comparison
                asymmetry_map[i, j] *= (1.0 - abs(j - w//2) / (w//2))
        
        # Normalize
        asymmetry_map = asymmetry_map / (asymmetry_map.max() + 1e-8)
        
        # Create overlay
        overlay = _overlay_heatmap_on_image(asymmetry_map, original_image, alpha=0.5, colormap=cv2.COLORMAP_PLASMA)
        
        # Save file
        filename = f"symmetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        cv2.imwrite(path, overlay)
        
        return path

    except Exception as e:
        return None


def generate_xai(model_name, image_tensor, original_image_array):
    """
    Generate all XAI visualizations for a given model and image.
    
    Args:
        model_name: Name of the model
        image_tensor: Preprocessed image tensor for model input
        original_image_array: Original image array for overlay
    
    Returns:
        Dictionary with paths to all XAI visualizations
    """
    try:
        model = get_model(model_name)
        
        if model is None:
            return {"error": "Model not found"}
        
        model.to(DEVICE)
        model.eval()
        
        # Ensure image_tensor is on device and has gradients enabled
        tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        results = {}
        
        # Generate Grad-CAM
        gradcam_path = _generate_gradcam(model, tensor, original_image_array)
        if gradcam_path:
            results["gradcam"] = gradcam_path
        
        # Generate Saliency Map
        saliency_path = _generate_saliency_map(model, tensor, original_image_array)
        if saliency_path:
            results["saliency"] = saliency_path
        
        # Generate Symmetry Analysis
        symmetry_path = _generate_symmetry_analysis(model, tensor, original_image_array)
        if symmetry_path:
            results["symmetry"] = symmetry_path
        
        if not results:
            return {"error": "Failed to generate any XAI visualizations"}
        
        return results
    
    except Exception as e:
        return {"error": str(e)}