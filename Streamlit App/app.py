import streamlit as st
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model


# Load the pretrained model
pretrained_model = load_model('best_model.h5')

# Compile the model manually
pretrained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the emotion labels (assuming they are the same as used in training)
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(uploaded_file, target_size=(48, 48)):
    # Load the image
    img = Image.open(uploaded_file)
    img = img.convert('L')  # Convert the image to grayscale
    img = img.resize(target_size)  # Resize the image to the target size
    img_array = img_to_array(img)
    normalize=img_array/255.0
    reshaped=np.reshape(normalize,(1,48,48,1))
    return reshaped


if uploaded_file is not None:
    
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)

    # Check the input shape
    st.write(f"Image shape: {img_array.shape}")

    # Make predictions
    #predictions = np.argmax(pretrained_model.predict(tf.expand_dims(img_array, axis=0), verbose=0))
    predictions=pretrained_model.predict(img_array)
    label=np.argmax(predictions)

    # Display the uploaded image and prediction results
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Emotion: {emotion_labels[label]}")
   
    
   
    
   
