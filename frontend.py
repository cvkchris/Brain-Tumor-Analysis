import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
import tensorflow as tf
import Morphological_Ops.ipynb as shit

shit.kirees()

# Load the saved model
model = load_model('Brain_Tumor_cnn.h5')

# Define the class mapping
class_mapping = {
    0: "Glioma tumor",
    1: "No tumor",
    2: "Pituitary tumor",
    3: "Meningioma tumor"
}

# Function to predict the class for an input image
def predict_class(image):
    # Resize the image to (224, 224) and preprocess it
    img = resize(image, (224, 224)) / 255.0
    img = img.numpy()
    img = img.reshape(1, 224, 224, 3)

    # Make predictions using the model
    predictions = model.predict(img)

    # Get the predicted class label
    predicted_label = class_mapping[predictions.argmax(axis=1)[0]]

    return predicted_label
st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.pexels.com/photos/15848946/pexels-photo-15848946/free-photo-of-gradient-graphic-design.jpeg");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.title("Brain Tumor Classification")
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    try:
        # Read the uploaded image
        img = tf.image.decode_image(uploaded_image.read(), channels=3)
        

        # Display the uploaded image
        st.image(img.numpy(), caption = "Uploaded Image",width = 300)
        
        predicted_class = predict_class(img)
        st.subheader("Result")
        st.info(f"The predicted class for the image is: {predicted_class}")

    except Exception as e:
        st.text(f"An error occurred: {e}")
