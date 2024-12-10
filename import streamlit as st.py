import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image as Image

# Load the pre-trained model (for example, a CNN model trained on CIFAR-10)
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# CIFAR-10 class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit interface
st.title("CIFAR-10 Image Classification")
st.write("Upload an image from CIFAR-10 dataset or any image to classify it into one of the 10 categories.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare the image for prediction
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize the image to match model's input range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the image class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display prediction result
    st.write(f"Predicted class: {labels[predicted_class]}")

    # Show the class probability
    class_probabilities = tf.nn.softmax(prediction[0])
    st.write(f"Class probabilities: {dict(zip(labels, class_probabilities))}")