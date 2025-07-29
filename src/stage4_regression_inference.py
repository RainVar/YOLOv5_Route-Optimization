import csv
import joblib
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
REGRESSION_MODEL_PATH = "../models/paser_regressor.joblib"  # Path to regression model
DETECTIONS_CSV = "detections.csv"                        # Input: YOLOv5 detections
SCORES_CSV = "proxy_paser_scores.csv"                    # Output: PASER scores per image

feature_columns = [
    "Alligator crack_count", "Alligator crack_total_area",
    "Longitudinal crack_count", "Longitudinal crack_total_area",
    "Longitudinal patch_count", "Longitudinal patch_total_area",
    "Pothole_count", "Pothole_total_area",
    "Transverse crack_count", "Transverse crack_total_area",
    "Transverse patch_count", "Transverse patch_total_area",
    "Manhole cover_count", "Manhole cover_total_area"
]

class_map = {
    0: ("Alligator crack_count", "Alligator crack_total_area"),
    1: ("Longitudinal crack_count", "Longitudinal crack_total_area"),
    2: ("Longitudinal patch_count", "Longitudinal patch_total_area"),
    3: ("Pothole_count", "Pothole_total_area"),
    4: ("Transverse crack_count", "Transverse crack_total_area"),
    5: ("Transverse patch_count", "Transverse patch_total_area"),
    6: ("Manhole cover_count", "Manhole cover_total_area"),
}

# Load regression model
regressor = joblib.load(REGRESSION_MODEL_PATH)

# Read detections and group by image
# Each image may have multiple detections
# We'll aggregate features for each image

detections_by_image = {}
with open(DETECTIONS_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_path = row["image_path"]
        if image_path not in detections_by_image:
            detections_by_image[image_path] = []
        detections_by_image[image_path].append(row)

# Prepare output CSV
SCORES_HEADER = [
    "segment_id", "u", "v", "k", "index", "lat", "lng", "heading", "image_path", "proxy_paser_score"
]
with open(SCORES_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(SCORES_HEADER)

    for image_path, detections in detections_by_image.items():
        # Initialize features
        features = {col: 0 for col in feature_columns}
        for det in detections:
            cls = int(det["class"])
            area = (float(det["xmax"]) - float(det["xmin"])) * (float(det["ymax"]) - float(det["ymin"]))
            count_col, area_col = class_map.get(cls, (None, None))
            if count_col and area_col:
                features[count_col] += 1
                features[area_col] += area
        # Prepare feature vector in correct order
        feature_vector = [features[col] for col in feature_columns]
        try:
            # Predict proxy PASER score for this image
            score = regressor.predict([feature_vector])[0]
        except Exception as e:
            print(f"Error predicting PASER score for {image_path}: {e}")
            continue
        # Use metadata from first detection (all detections for image share metadata)
        meta = detections[0]
        writer.writerow([
            meta["segment_id"], meta["u"], meta["v"], meta["k"], meta["index"],
            meta["lat"], meta["lng"], meta["heading"], meta["image_path"], score
        ])