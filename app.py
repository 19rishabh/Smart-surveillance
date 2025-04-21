# app.py - Integration with your existing models
from flask import Flask, render_template, Response, jsonify
import cv2
import time
import json
from datetime import datetime
from collections import defaultdict
import threading
import os

# Import your custom detection modules
from overspeeding import OverspeedingDetector
from ultralytics import YOLO  # For weapon/fire detection

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
detection_data = {
    "vehicles": [],
    "weapons": 0,
    "fires": 0,
    "speeds": [],
    "alerts": []
}
processing_thread = None
running = False

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Use 0 for webcam or RTSP URL for CCTV
        if not camera.isOpened():
            print("ERROR: Could not open camera")
            return None
    return camera

def process_surveillance():
    """Process surveillance using your existing models"""
    global output_frame, detection_data, lock, running
    
    # Get camera reference
    cam = get_camera()
    if cam is None:
        print("Failed to get camera")
        return
    
    # Set running flag
    running = True
    
    # Initialize your models
    try:
        print("Loading object detection model...")
        weapon_fire_model = YOLO("runs/detect/train/weights/best.pt")
        
        print("Loading overspeeding detection model...")
        speed_detector = OverspeedingDetector(
            video_path=0,  # Use webcam
            model_path="yolov8n.pt",
            speed_limit=30
        )
        
        models_loaded = True
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        models_loaded = False
    
    print("Starting surveillance processing loop...")
    while running:
        success, frame = cam.read()
        if not success:
            print("Failed to read frame")
            time.sleep(0.1)
            continue
        
        processed_frame = frame.copy()
        
        # Process with your models if loaded
        if models_loaded:
            try:
                # 1. Weapon/Fire detection
                weapon_results = weapon_fire_model(frame, conf=0.35)
                
                if weapon_results and len(weapon_results) > 0:
                    # Draw results on the frame
                    processed_frame = weapon_results[0].plot()
                    
                    # Process detections
                    weapon_count = 0
                    fire_count = 0
                    
                    for box in weapon_results[0].boxes:
                        conf = float(box.conf[0].item())
                        if conf < 0.5:
                            continue  # Skip low-confidence detections
                        cls_id = int(box.cls[0].item())
                        cls_name = weapon_results[0].names[cls_id]
                        
                        # Assuming class names or IDs for weapons/fire
                        # Adjust these based on your actual model classes
                        if "weapon" in cls_name.lower() or "gun" in cls_name.lower() or cls_name.lower() == "knife":
                            weapon_count += 1
                            # Create alert for weapons
                            with lock:
                                detection_data["alerts"].append({
                                    "type": "WEAPON DETECTED",
                                    "details": f"{cls_name} detected with {conf:.2f} confidence",
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                        
                        if "fire" in cls_name.lower() or "smoke" in cls_name.lower():
                            fire_count += 1
                            # Create alert for fire
                            with lock:
                                detection_data["alerts"].append({
                                    "type": "FIRE/SMOKE DETECTED",
                                    "details": f"{cls_name} detected with {conf:.2f} confidence",
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                    
                    # Update detection data
                    with lock:
                        detection_data["weapons"] = weapon_count
                        detection_data["fires"] = fire_count
                
                # 2. Speed detection
                speed_frame = speed_detector.process_frame(processed_frame.copy())
                
                # Use the speed detector's processed frame as it has speed overlays
                processed_frame = speed_frame
                
                # Extract speed data from speed detector
                # This depends on how your OverspeedingDetector is implemented
                # Here we're assuming it has car_tracks and line_crossings
                with lock:
                    if hasattr(speed_detector, 'car_tracks'):
                        detection_data["vehicles"] = list(speed_detector.car_tracks.keys())
                    
                    # Add speeding alerts
                    for car_id in detection_data["vehicles"]:
                        if hasattr(speed_detector, 'speed_history') and car_id in speed_detector.speed_history:
                            speed = speed_detector.speed_history[car_id][-1] if speed_detector.speed_history[car_id] else 0
                            
                            # Add overspeeding alerts
                            if speed > speed_detector.speed_limit:
                                detection_data["alerts"].append({
                                    "type": "OVERSPEEDING",
                                    "details": f"Vehicle ID {car_id} detected at {speed:.1f} km/h (limit: {speed_detector.speed_limit})",
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                
            except Exception as e:
                print(f"Error in surveillance processing: {e}")
                import traceback
                traceback.print_exc()
        
        # Keep only recent alerts
        with lock:
            if len(detection_data["alerts"]) > 10:
                detection_data["alerts"] = detection_data["alerts"][-10:]
        
        # Add timestamp to frame
        cv2.putText(processed_frame, 
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update output frame
        with lock:
            output_frame = processed_frame.copy()
        
        # Brief pause
        time.sleep(0.01)

def generate_frames():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.1)
                continue
            
            # Encode the frame as JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if not ret:
                    time.sleep(0.1)
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"Error encoding frame: {e}")
                time.sleep(0.1)
                continue
        
        # Yield the frame in the MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection_data')
def get_detection_data():
    global detection_data
    with lock:
        # Create a copy to avoid threading issues
        data_copy = {
            "vehicles": len(detection_data["vehicles"]),
            "weapons": detection_data["weapons"],
            "fires": detection_data["fires"],
            "alerts": detection_data["alerts"].copy() if detection_data["alerts"] else []
        }
    return jsonify(data_copy)

@app.route('/api/toggle_detection')
def toggle_detection():
    global running, processing_thread
    
    if running:
        # Stop detection
        running = False
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)
        return jsonify({"status": "stopped"})
    else:
        # Start detection
        running = True
        processing_thread = threading.Thread(target=process_surveillance)
        processing_thread.daemon = True
        processing_thread.start()
        return jsonify({"status": "running"})

if __name__ == '__main__':
    try:
        # Initialize the camera before starting the app
        if get_camera() is not None:
            # Start processing thread
            processing_thread = threading.Thread(target=process_surveillance)
            processing_thread.daemon = True
            processing_thread.start()
            
            print("Starting Flask server...")
            # Start the Flask app
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        else:
            print("Error: Could not initialize camera.")
    except KeyboardInterrupt:
        print("Exiting...")
        running = False
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)