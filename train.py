from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO)
model = YOLO("yolov8n.pt")  # You can use yolov8m.pt or yolov8x.pt for larger models

# Train the model on your dataset
model.train(data="dataset/data.yaml", epochs=20, batch=16, imgsz=640)

