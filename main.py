import cv2  # Importing OpenCV for computer vision tasks
import streamlit as st  # Importing Streamlit for building interactive web applications
import numpy as np  # Importing NumPy for numerical computing
from PIL import Image  # Importing PIL for image processing
import face_recognition  # Importing face_recognition for face identification

# Function to detect faces and recognize known faces in the live camera stream
def detect_faces():
    # Load the Haar cascade for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Placeholder for displaying the video feed in Streamlit
    stframe = st.empty()

    # Use session state to control camera loop
    if "stop_camera" not in st.session_state:
        st.session_state.stop_camera = False

    # Create a button to stop the camera feed
    stop_button = st.button("Stop Camera", key="stop_camera_button")

    # Load known faces and names (For example purposes, use one image of a known face)
    known_face_encodings = []
    known_face_names = []

    # Example: Add known face (in real use, add multiple known faces)
    img_of_person = face_recognition.load_image_file(r"C:\JASWAN\fr\images (1).jpeg")  # Use raw string literal for Windows path
    img_encoding = face_recognition.face_encodings(img_of_person)[0]  # Get face encoding
    known_face_encodings.append(img_encoding)
    known_face_names.append("Dhoni")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the camera")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Find face locations and encodings for known face recognition
        face_locations = []
        face_encodings = []
        
        for (x, y, w, h) in faces:
            # Crop the face area from the frame
            face = frame[y:y+h, x:x+w]

            # Convert the face image to RGB (required by face_recognition)
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Get face encoding for the current face using face_recognition
            face_encoding = face_recognition.face_encodings(rgb_face)
            if face_encoding:
                face_encodings.append(face_encoding[0])
                face_locations.append((x, y, x + w, y + h))

        # Compare faces and recognize if they match any known faces
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

        # Draw rectangles and display names for recognized faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the video stream in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Stop the camera feed if the "Stop Camera" button is clicked
        if stop_button:
            st.session_state.stop_camera = True
            break

    cap.release()

# Function to detect faces in an uploaded image
def detect_faces_in_image(uploaded_image):
    # Convert the uploaded image file to a NumPy array
    img_array = np.array(Image.open(uploaded_image))

    # Create the Haar cascade for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Parameter specifying how much the image size is reduced at each image scale
        minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this will be ignored
    )

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting image with face detection
    st.image(img_array, caption="Detected Faces", use_container_width=True)

# ================================= Streamlit App ======================================
# Streamlit UI
st.title("Face Detection and Recognition")
st.subheader("Either open the camera or upload an image to detect and recognize faces.")

# Button to start face detection in live camera stream
if st.button("Open Camera"):
    detect_faces()

# File uploader for detecting faces in an uploaded image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
