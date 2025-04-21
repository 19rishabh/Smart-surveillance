import cv2
import time
import numpy as np
import csv
from collections import deque
from ultralytics import YOLO

class OverspeedingDetector:
    def __init__(self, video_path, model_path="yolov8n.pt", speed_limit=25):
        # Core components
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Speed calculation parameters - increased for high-angle camera
        self.scale_factor = 0.7  # Increased from 0.035 - needs proper calibration
        self.speed_limit = speed_limit  # km/h
        self.smooth_factor = 0.7  # Slightly reduced for faster response to speed changes
        
        # Advanced tracking
        self.car_tracks = {}  # Track each car's movement
        self.speed_history = {}  # Store speed history for averaging
        self.history_size = 3  # Reduced from 5 to be more responsive
        
        # Calibration lines (moved to middle of frame)
        mid_y = int(self.frame_height * 0.5)  # Center of frame
        upper_y = int(self.frame_height * 0.25)  # Above center
        lower_y = int(self.frame_height * 0.45)  # Below center
        
        self.detection_lines = [
            # Format: [(x1, y1), (x2, y2), name]
            [(int(self.frame_width * 0.1), upper_y), 
             (int(self.frame_width * 0.9), upper_y), "Line1"],
            [(int(self.frame_width * 0.1), lower_y), 
             (int(self.frame_width * 0.9), lower_y), "Line2"]
        ]
        
        # Increased real-world distance between lines to account for camera angle
        self.line_distance_meters = 10  # Increased from 5 meters based on perspective
        self.line_crossings = {}  # Track when objects cross lines
        
        # Speed correction for perspective (vehicles further from camera appear slower)
        self.perspective_correction = True
        self.camera_height_factor = 1.5  # Adjust based on camera height
        
        # Results logging
        self.log_file = open("car_speeds.csv", "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["Car ID", "Speed (km/h)", "Timestamp", "Status"])
        
        # Visualization settings
        self.show_visualization = True
        self.show_debug_info = True  # Display extra debugging info
        self.blur_faces = True  # Privacy protection

    def calculate_cross_time(self, car_id, line_name, current_time):
        """Record when a car crosses a detection line"""
        if car_id not in self.line_crossings:
            self.line_crossings[car_id] = {}
        
        self.line_crossings[car_id][line_name] = current_time
        
    def has_crossed_line(self, previous_pos, current_pos, line):
        """Check if object has crossed a line between frames"""
        line_start, line_end, _ = line
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Check if line segments intersect
        A, B = previous_pos, current_pos
        C, D = line_start, line_end
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def apply_perspective_correction(self, speed, y_pos):
        """Correct speed based on perspective (objects further from camera appear slower)"""
        if not self.perspective_correction:
            return speed
            
        # Calculate distance from bottom of frame (closer to camera)
        distance_from_bottom = self.frame_height - y_pos
        
        # Apply correction factor (higher values for objects higher in frame)
        correction = 1.0 + (distance_from_bottom / self.frame_height) * self.camera_height_factor
        
        return speed * correction
    
    def calculate_speed(self, car_id, y_pos):
        """Calculate speed based on time between crossing detection lines"""
        if car_id in self.line_crossings:
            crossings = self.line_crossings[car_id]
            if "Line1" in crossings and "Line2" in crossings:
                time_diff = abs(crossings["Line2"] - crossings["Line1"])
                if time_diff > 0:
                    # Calculate speed in km/h
                    speed = (self.line_distance_meters / time_diff) * 3.6
                    
                    # Apply perspective correction
                    speed = self.apply_perspective_correction(speed, y_pos)
                    
                    # Store in speed history for averaging
                    if car_id not in self.speed_history:
                        self.speed_history[car_id] = deque(maxlen=self.history_size)
                    self.speed_history[car_id].append(speed)
                    
                    # Return average speed over history
                    return np.mean(self.speed_history[car_id])
        
        return None
    
    def process_frame(self, frame):
        """Process a single video frame"""
        # Run YOLOv8 detection with tracking
        results = self.model.track(frame, persist=True, conf=0.45, iou=0.45, classes=[2, 3, 5, 7])  # Car, Motorcycle, Bus, Truck
        
        current_time = time.time()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get tracked objects
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
                
                for i, (x, y, w, h) in enumerate(boxes):
                    x_center, y_center = int(x), int(y)
                    car_id = track_ids[i]
                    
                    # Store center position
                    current_pos = (x_center, y_center)
                    
                    # Check if car has crossed any detection line
                    if car_id in self.car_tracks:
                        prev_pos = (self.car_tracks[car_id][0], self.car_tracks[car_id][1])
                        
                        for line in self.detection_lines:
                            if self.has_crossed_line(prev_pos, current_pos, line):
                                self.calculate_cross_time(car_id, line[2], current_time)
                    
                    # Update car position
                    self.car_tracks[car_id] = (x_center, y_center, current_time)
                    
                    # Calculate speed if car has crossed both lines
                    speed = self.calculate_speed(car_id, y_center)
                    
                    if speed is not None:
                        # Determine status (overspeeding or normal)
                        status = "OVERSPEEDING" if speed > self.speed_limit else "NORMAL"
                        color = (0, 0, 255) if status == "OVERSPEEDING" else (0, 255, 0)
                        
                        # Log to CSV
                        self.csv_writer.writerow([car_id, round(speed, 1), 
                                                  time.strftime("%H:%M:%S"), status])
                        
                        # Display on frame
                        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), 
                                      (int(x + w/2), int(y + h/2)), color, 2)
                        cv2.putText(frame, f"ID: {car_id}", (int(x - w/2), int(y - h/2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(frame, f"{round(speed, 1)} km/h", (int(x - w/2), int(y - h/2) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw detection lines
        for line_start, line_end, line_name in self.detection_lines:
            cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
            cv2.putText(frame, line_name, line_start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add speed limit indicator
        cv2.putText(frame, f"Speed Limit: {self.speed_limit} km/h", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add debug info
        if self.show_debug_info:
            cv2.putText(frame, f"Line distance: {self.line_distance_meters}m", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Perspective correction: {self.perspective_correction}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            
            if self.show_visualization:
                cv2.imshow("Overspeeding Detection System", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.log_file.close()
        
    def calibrate(self, known_distance_meters=None, known_speed_kmh=None):
        """Calibrate using known real-world distance or speed"""
        if known_distance_meters is not None:
            self.line_distance_meters = known_distance_meters
            print(f"Set line distance to {known_distance_meters} meters")
            
        if known_speed_kmh is not None:
            # Adjust scale factor based on known speed
            # This requires manual calibration with a vehicle of known speed
            self.known_calibration_speed = known_speed_kmh
            print(f"Set calibration speed to {known_speed_kmh} km/h")
            
        return True


# Run with adjusted parameters for higher-angle camera
if __name__ == "__main__":
    detector = OverspeedingDetector(
        video_path="highway3.mp4",
        model_path="yolov8n.pt",
        speed_limit=80
    )
    
    # Calibrate for high-angle camera (adjust these values based on your setup)
    detector.calibrate(known_distance_meters=17)  # Increased from default
    detector.camera_height_factor = 1.3  # correction factor
    
    # Run the detection
    detector.run()