import face_recognition
import cv2
import numpy as np
import os

# Load known face encodings and names from the "known_faces" folder
known_faces_folder = "known_faces"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(known_faces_folder, filename)

        # Load image and get encodings
        known_image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(known_image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Name from filename

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (face_recognition expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        name = "Unknown"  # Default label

        if best_match_index is not None and matches[best_match_index]:
            name = "Criminal"  # Label only known faces as "Criminal"

        # Draw bounding box and label
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video feed
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
