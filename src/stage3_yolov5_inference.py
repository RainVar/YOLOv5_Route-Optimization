import os
import csv
import torch
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "models/best.pt"           # Path to YOLOv5 model
IMAGE_METADATA = "data/image_metadata.csv"   # Metadata CSV from image download stage
DETECTIONS_CSV = "data/detections.csv"       # Output CSV for detections

# Load YOLOv5 model (from torch hub)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# Read image metadata
with open(IMAGE_METADATA, newline='') as f:
    reader = csv.DictReader(f)
    image_rows = list(reader)

# Prepare detections CSV
DETECTION_HEADER = [
    "segment_id", "u", "v", "k", "index", "lat", "lng", "heading", "image_path",
    "class", "confidence", "xmin", "ymin", "xmax", "ymax"
]
with open(DETECTIONS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(DETECTION_HEADER)

    for row in image_rows:
        image_path = row["image_path"]
        # Convert to absolute path relative to project root
        abs_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", image_path.replace("\\", os.sep)))
        if not os.path.exists(abs_image_path):
            print(f"Image not found: {abs_image_path}")
            continue
        try:
            # Run YOLOv5 inference
            results = model(abs_image_path)
            detections = results.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax, conf, cls]
            for det in detections:
                xmin, ymin, xmax, ymax, conf, cls = det
                writer.writerow([
                    row["segment_id"], row["u"], row["v"], row["k"], row["index"],
                    row["lat"], row["lng"], row["heading"], row["image_path"],
                    int(cls), float(conf), float(xmin), float(ymin), float(xmax), float(ymax)
                ])
        except Exception as e:
            print(f"Error running YOLOv5 on {abs_image_path}: {e}")
            continue