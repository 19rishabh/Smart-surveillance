from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Path to the best trained model

# Test on a single image
results = model("gun.jpg", save=True, conf=0.4)  # Replace "test.jpg" with your test image

# Show the results
print(results)  # Print detected objects



