import io
import os
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchvision
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_faster_rcnn_model():
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    except (ImportError, AttributeError):
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    # Check for fine-tuned weights
    weight_candidates = [
        "artifacts/models/fasterrcnn.pth",
        "artifacts/models/fasterrcbb.pth",
        os.path.join(os.path.dirname(__file__), "artifacts", "models", "fasterrcnn.pth")
    ]
    for wpath in weight_candidates:
        if os.path.exists(wpath):
            try:
                model.load_state_dict(torch.load(wpath, map_location=device))
                break
            except Exception:
                pass
                
    model.to(device)
    model.eval()
    return model

model = load_faster_rcnn_model()

transform = transforms.Compose([
    transforms.ToTensor()
])

app = FastAPI(
    title="Guns Object Detection Studio",
    description="Real-time Object Detection API with Faster R-CNN and FastAPI",
    version="1.0.0"
)

# Ensure static & templates directory exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def filter_detections_with_nms(prediction, confidence_threshold=0.60, iou_threshold=0.30):
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']
    
    # 1. Filter by confidence threshold
    keep_mask = (scores >= confidence_threshold) & (labels == 1)
    filtered_boxes = boxes[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_labels = labels[keep_mask]
    
    if len(filtered_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # 2. Apply Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes
    keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold=iou_threshold)
    
    final_boxes = filtered_boxes[keep_indices].cpu().numpy()
    final_scores = filtered_scores[keep_indices].cpu().numpy()
    final_labels = filtered_labels[keep_indices].cpu().numpy()
    
    return final_boxes, final_scores, final_labels

def predict_and_draw(image: Image.Image, confidence_threshold: float = 0.60, iou_threshold: float = 0.30) -> Image.Image:
    img_rgb = image.convert("RGB")
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
        
    prediction = predictions[0]
    boxes, scores, labels = filter_detections_with_nms(
        prediction, 
        confidence_threshold=confidence_threshold, 
        iou_threshold=iou_threshold
    )
    
    draw = ImageDraw.Draw(img_rgb)
    
    if len(boxes) > 0:
        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            # Bold high-contrast red bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline='#ef4444', width=3)
            
            # Badge header for label
            label_text = f"Gun: {int(score * 100)}%"
            text_bbox = draw.textbbox((x_min, max(0, y_min - 20)), label_text)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill='#ef4444')
            draw.text((x_min, max(0, y_min - 20)), label_text, fill='#ffffff')
            
    return img_rgb

def detect_objects_json(image: Image.Image, confidence_threshold: float = 0.60, iou_threshold: float = 0.30):
    img_rgb = image.convert("RGB")
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)
    latency_ms = round((time.time() - start_time) * 1000, 1)
        
    prediction = predictions[0]
    boxes, scores, labels = filter_detections_with_nms(
        prediction, 
        confidence_threshold=confidence_threshold, 
        iou_threshold=iou_threshold
    )
    
    filtered_boxes = []
    max_score = 0.0
    
    if len(boxes) > 0:
        for box, score in zip(boxes, scores):
            max_score = max(max_score, float(score))
            filtered_boxes.append({
                "box": [float(b) for b in box],
                "score": float(score),
                "label": "Gun"
            })
            
    return {
        "count": len(filtered_boxes),
        "detections": filtered_boxes,
        "max_score": max_score,
        "latency_ms": latency_ms,
        "device": str(device)
    }

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Guns Detection API</h1><p>Template index.html not found</p>")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), confidence: float = 0.60, iou: float = 0.30):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        output_image = predict_and_draw(image, confidence_threshold=confidence, iou_threshold=iou)
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/detect/")
async def api_detect(file: UploadFile = File(...), confidence: float = 0.60, iou: float = 0.30):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        result = detect_objects_json(image, confidence_threshold=confidence, iou_threshold=iou)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/api/info")
def get_info():
    return {
        "status": "online",
        "model": "Faster R-CNN ResNet50-FPN",
        "classes": ["Background", "Gun"],
        "device": str(device),
        "weights_loaded": os.path.exists("artifacts/models/fasterrcnn.pth")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)