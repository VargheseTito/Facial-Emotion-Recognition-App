import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model
import cv2

# Load the pretrained model
pretrained_model = load_model('best_model.h5')

# Compile the model manually if required
# pretrained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the emotion labels (assuming they are the same as used in training)
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize video capture
video = cv2.VideoCapture(0)

# Load the Haar cascade for face detection
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        # Extract the face region from the image
        sub_face_img = gray[y:y+h, x:x+w]
        # Resize the face image to match the model input size
        resized = cv2.resize(sub_face_img, (48, 48))
        # Normalize the resized face image
        normalize = resized / 255.0
        # Reshape the normalized image to match the model input shape
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        # Make predictions using the pre-trained model
        result = pretrained_model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, emotion_labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
video.release()
cv2.destroyAllWindows()
