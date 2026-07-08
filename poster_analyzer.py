import os
import sys
import json
import cv2
import numpy as np

# Try importing torch and torchvision for semantic object detection
HAS_TORCH = False
try:
    import torch
    import torchvision
    from torchvision.models.detection import (
        ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    )
    HAS_TORCH = True
except Exception as e:
    sys.stderr.write(f"[poster_analyzer] Warning: Failed to import PyTorch/Torchvision. Object detection disabled. Error: {e}\n")

def run_opencv_layout(img):
    """
    Detect text blocks, titles, and graphical regions using OpenCV morphological operations and contour analysis.
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle various lighting conditions on posters
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Dynamic kernel sizes relative to image dimensions
    kw_h = max(5, int(w * 0.025))
    kh_h = max(1, int(h * 0.003))
    kw_v = max(1, int(w * 0.003))
    kh_v = max(5, int(h * 0.015))
    
    # Dilate horizontally to merge letters into words and lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_h, kh_h))
    dilated = cv2.dilate(thresh, kernel_h, iterations=1)
    
    # Dilate vertically to merge lines into block/paragraphs
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_v, kh_v))
    dilated = cv2.dilate(dilated, kernel_v, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elements = []
    text_count = 0
    title_count = 0
    visual_count = 0
    
    # Filter and classify contours
    min_area = (w * h) * 0.0005 # At least 0.05% of image area
    for idx, c in enumerate(contours):
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area:
            continue
            
        # Ignore extremely large regions (e.g. background borders)
        if bw > w * 0.95 and bh > h * 0.95:
            continue
            
        # Calculate aspect ratio
        ar = bw / float(bh)
        
        # Normalized coordinates (0.0 to 1.0)
        nx = bx / float(w)
        ny = by / float(h)
        nw = bw / float(w)
        nh = bh / float(h)
        
        # Classification heuristics
        if ny < 0.35 and bw > w * 0.3 and bh > h * 0.03 and ar > 1.5:
            title_count += 1
            label = f"Title/Headline {title_count}"
            el_type = "title"
        elif bw > w * 0.15 and bh > h * 0.02:
            text_count += 1
            label = f"Text Block {text_count}"
            el_type = "text"
        else:
            visual_count += 1
            label = f"Visual Block {visual_count}"
            el_type = "layout"
            
        elements.append({
            "label": label,
            "type": el_type,
            "box": [nx, ny, nw, nh],
            "score": 0.85 # Heuristic confidence
        })
        
    return elements

def run_torchvision_objects(img):
    """
    Detect semantic objects (like people, cup, handbag, etc.) using PyTorch models.
    Tries Faster R-CNN ResNet-50 first, then Faster R-CNN MobileNet, then SSDLite.
    """
    if not HAS_TORCH:
        return []
        
    h, w, _ = img.shape
    model = None
    weights = None
    
    # Cascade loading
    # 1. Faster R-CNN ResNet-50 (Highest accuracy)
    try:
        sys.stderr.write("[poster_analyzer] Attempting to load Faster R-CNN ResNet-50...\n")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        sys.stderr.write("[poster_analyzer] Loaded Faster R-CNN ResNet-50 successfully.\n")
    except Exception as e:
        sys.stderr.write(f"[poster_analyzer] Faster R-CNN ResNet-50 load failed: {e}. Trying MobileNet FPN...\n")
        
    # 2. Faster R-CNN MobileNet V3 Large FPN (Medium size, high accuracy)
    if model is None:
        try:
            sys.stderr.write("[poster_analyzer] Attempting to load Faster R-CNN MobileNet FPN...\n")
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            sys.stderr.write("[poster_analyzer] Loaded Faster R-CNN MobileNet FPN successfully.\n")
        except Exception as e:
            sys.stderr.write(f"[poster_analyzer] Faster R-CNN MobileNet FPN load failed: {e}. Trying SSDLite...\n")
            
    # 3. SSDLite MobileNet V3 Large (Lightweight fallback)
    if model is None:
        try:
            sys.stderr.write("[poster_analyzer] Attempting to load SSDLite MobileNet...\n")
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = ssdlite320_mobilenet_v3_large(weights=weights)
            sys.stderr.write("[poster_analyzer] Loaded SSDLite successfully.\n")
        except Exception as e:
            sys.stderr.write(f"[poster_analyzer] SSDLite load failed: {e}.\n")
            return []
            
    try:
        model.eval()
        preprocess = weights.transforms()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        input_tensor = preprocess(img_tensor)
        
        with torch.no_grad():
            predictions = model([input_tensor])
            
        preds = predictions[0]
        boxes = preds["boxes"].numpy()
        labels = preds["labels"].numpy()
        scores = preds["scores"].numpy()
        categories = weights.meta["categories"]
        
        elements = []
        counts = {}
        
        # Filter detections by confidence score
        for i in range(len(scores)):
            score = float(scores[i])
            if score < 0.35:
                continue
                
            label_idx = int(labels[i])
            label_name = categories[label_idx]
            
            bx1, by1, bx2, by2 = boxes[i]
            
            # Clamp boxes to image boundary
            bx1 = max(0.0, min(bx1, float(w)))
            by1 = max(0.0, min(by1, float(h)))
            bx2 = max(0.0, min(bx2, float(w)))
            by2 = max(0.0, min(by2, float(h)))
            
            bw = bx2 - bx1
            bh = by2 - by1
            
            if bw < 10 or bh < 10:
                continue
                
            # Normalized coordinates
            nx = bx1 / float(w)
            ny = by1 / float(h)
            nw = bw / float(w)
            nh = bh / float(h)
            
            counts[label_name] = counts.get(label_name, 0) + 1
            full_label = f"{label_name.capitalize()} {counts[label_name]}"
            
            elements.append({
                "label": full_label,
                "type": "object",
                "box": [nx, ny, nw, nh],
                "score": score
            })
            
        return elements
    except Exception as e:
        sys.stderr.write(f"[poster_analyzer] Object detection execution failed: {e}\n")
        return []

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing image path argument"}))
        return
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(json.dumps({"success": False, "error": f"Image file not found: {img_path}"}))
        return
        
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(json.dumps({"success": False, "error": f"Failed to decode image: {img_path}"}))
            return
            
        # 1. Run OpenCV layout analysis
        cv_elements = run_opencv_layout(img)
        
        # 2. Run PyTorch object detection
        torch_elements = run_torchvision_objects(img)
        
        # 3. Merge elements
        all_elements = cv_elements + torch_elements
        
        result = {
            "success": True,
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "elements": all_elements
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Internal error during analysis: {e}"}))

if __name__ == "__main__":
    main()
