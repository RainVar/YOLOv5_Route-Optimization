import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DETECTIONS_CSV = os.path.join(os.path.dirname(__file__), '..', 'detections.csv')

def show_image_with_detections(image_path, detections):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for det in detections:
        xmin = float(det['xmin'])
        ymin = float(det['ymin'])
        xmax = float(det['xmax'])
        ymax = float(det['ymax'])
        cls = det['class']
        conf = float(det['confidence'])
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"Class {cls} ({conf:.2f})", color='yellow', fontsize=8, backgroundcolor='black')

    plt.title(os.path.basename(image_path))
    plt.axis('off')
    plt.show()

def main():
    # Group detections by image
    detections_by_image = {}
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    with open(DETECTIONS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['image_path'].replace("\\", os.sep)
            abs_img_path = os.path.abspath(os.path.join(project_root, img_path))
            if abs_img_path not in detections_by_image:
                detections_by_image[abs_img_path] = []
            detections_by_image[abs_img_path].append(row)

    # Show each image and its detections
    for img_path, dets in detections_by_image.items():
        if os.path.exists(img_path):
            print(f"Showing: {img_path}")
            show_image_with_detections(img_path, dets)
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    main()