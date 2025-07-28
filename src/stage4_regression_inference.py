import csv
import joblib

# -----------------------------
# CONFIGURATION
# -----------------------------
REGRESSION_MODEL_PATH = "models/paser_regressor.joblib"  # Path to regression model
DETECTIONS_CSV = "detections.csv"                        # Input: YOLOv5 detections
SCORES_CSV = "proxy_paser_scores.csv"                    # Output: PASER scores per image

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
        # Aggregate detection features for regression
        features = []
        for det in detections:
            # Use detection features relevant for regression
            features.extend([
                float(det["confidence"]),
                float(det["xmin"]),
                float(det["ymin"]),
                float(det["xmax"]),
                float(det["ymax"]),
                int(det["class"])
            ])
        try:
            # Predict proxy PASER score for this image
            score = regressor.predict([features])[0]
        except Exception as e:
            print(f"Error predicting PASER score for {image_path}: {e}")
            continue
        # Use metadata from first detection (all detections for image share metadata)
        meta = detections[0]
        writer.writerow([
            meta["segment_id"], meta["u"], meta["v"], meta["k"], meta["index"],
            meta["lat"], meta["lng"], meta["heading"], meta["image_path"], score
        ])