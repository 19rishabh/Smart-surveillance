import face_recognition
import pickle
import os

# Directory where the images of known faces are stored
known_faces_dir = "known_faces"

# Lists to hold the face encodings and names
known_face_encodings = []
known_face_names = []

# Loop through the known faces directory to encode images
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        # Load the image file
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        # Get the face encodings for the faces in the image
        encodings = face_recognition.face_encodings(image)

        # If at least one face encoding is found, use it
        if encodings:
            encoding = encodings[0]  # If there's more than one face, just pick the first one
            name = filename.split('.')[0]  # Use the file name (without extension) as the name

            # Append the encoding and the name to the lists
            known_face_encodings.append(encoding)
            known_face_names.append(name)

# Save the known faces and their encodings to a pickle file
with open('known_faces.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Faces preparation complete. Known faces saved in 'known_faces.pkl'.")
