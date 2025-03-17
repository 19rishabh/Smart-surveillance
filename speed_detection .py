import cv2
import pandas as pd
from ultralytics import YOLO
import os
import time
import math


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
# Load YOLO model
model = YOLO('yolov8s.pt')

# Load video file
cap = cv2.VideoCapture('highway.mp4')

# COCO class list
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

# Initialize tracker
tracker = Tracker()
count = 0
down, up = {}, {}
counter_down, counter_up = [], []

# Line positions for speed calculation
red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLO detection
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    detected_objects = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row[:6])
        if 0 <= d < len(class_list) and class_list[d] == 'car':
            detected_objects.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detected_objects)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Going Down
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()
        if id in down and blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            elapsed_time = time.time() - down[id]
            if id not in counter_down:
                counter_down.append(id)
                speed_kmh = (50 / elapsed_time) * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.putText(frame, str(int(speed_kmh)) + ' Km/h', (x4, y4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Going Up
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()
        if id in up and red_line_y < (cy + offset) and red_line_y > (cy - offset):
            elapsed1_time = time.time() - up[id]
            if id not in counter_up:
                counter_up.append(id)
                speed_kmh = (50 / elapsed1_time) * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.putText(frame, str(int(speed_kmh)) + ' Km/h', (x4, y4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw UI elements
    text_color = (0, 0, 0)
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.putText(frame, 'Red Line', (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
    cv2.putText(frame, 'Blue Line', (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    cv2.putText(frame, 'Going Down - ' + str(len(counter_down)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, 'Going Up - ' + str(len(counter_up)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Save frame and write to video
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)
    out.write(frame)

    # Display the frame
    cv2.imshow("Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
