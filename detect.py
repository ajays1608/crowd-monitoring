import cv2
from ultralytics import YOLO
import numpy as np

# Load the medium model (better for crowds than nano)
model = YOLO('yolov8m.pt')

def detect_persons(image_path):
    """
    HIGH ACCURACY person detection for massive crowds
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    h, w = image.shape[:2]
    
    # 1. Image Slicing / High-Res inference
    # We force the model to look at the image at a higher resolution
    # 2. Lower confidence to catch small/blurry people in the crowd
    # 3. Increase max_det so it doesn't stop counting at 50 people
    results = model.predict(
        image, 
        classes=[0],       # 0 = person class only
        conf=0.10,         # SUPER LOW CONFIDENCE for dense crowds!
        iou=0.45,          # Overlap threshold
        imgsz=1024,        # High resolution inference
        max_det=1000,      # Allow up to 1000 people to be detected
        verbose=False
    )
    
    # Create a copy of the image to draw boxes on
    annotated = image.copy()
    person_coords = []
    
    boxes = results[0].boxes
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            
            # Calculate center point for the heatmap
            cx = (x1 + x2) // 2
            cy = y2  # Use the bottom of the box (where their feet are)
            person_coords.append((cx, cy))
            
            # Draw a thin, clean green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Optional: Draw a small dot at their feet
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

    person_count = len(person_coords)
    
    # Dynamic Limit calculation (for database logging)
    # Estimate a safe limit based on image size and average person size
    if person_count > 0:
        areas = [(b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]) for b in boxes]
        median_area = float(np.median([a.cpu().numpy() for a in areas]))
        dynamic_limit = int((w * h * 0.6) / (median_area * 2.0)) if median_area > 0 else 100
    else:
        dynamic_limit = 100

    return annotated, person_count, person_coords, dynamic_limit
