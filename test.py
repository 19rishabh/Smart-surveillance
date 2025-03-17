from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Path to the best trained model

# Test on a single image
results = model("fire.jpg", save=True, conf=0.35)  # Replace "test.jpg" with your test image

# Show the results
print(results)  # Print detected objects



