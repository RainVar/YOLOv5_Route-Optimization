import os
import csv
import torch
from pathlib import Path

# Model file path
MODEL_PATH = "models/best.pt"
IMAGE_METADATA = "image_metadata.csv"
DETECTIONS_CSV = "detections.csv"

# Load YOLOv5 model
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
        if not os.path.exists(image_path):
            continue

        # Run YOLOv5 inference
        results = model(image_path)
        detections = results.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax, conf, cls]

        for det in detections:
            xmin, ymin, xmax, ymax, conf, cls = det
            writer.writerow([
                row["segment_id"], row["u"], row["v"], row["k"], row["index"],
                row["lat"], row["lng"], row["heading"], row["image_path"],
                int(cls), float(conf), float(xmin), float(ymin), float(xmax), float(ymax)
            ])