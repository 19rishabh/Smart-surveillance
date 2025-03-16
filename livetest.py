import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)  # 0 for default webcam, or use 'rtsp://...' for CCTV

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, conf=0.4)

    # Draw bounding boxes on the frame
    frame = results[0].plot()

    # Display the results
    cv2.imshow("Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
