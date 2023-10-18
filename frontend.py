import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
import tensorflow as tf
import cv2

def equalize_image(image):
    equalized = cv2.equalizeHist(image)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def morph_ops(image, kernel):
    # Apply binary threshold to the image
    _, gray_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # Apply morphological operations on the binary image
    closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

    return closed_image, opened_image, eroded_image, dilated_image

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

choice = st.sidebar.selectbox("Choose the Operation",('Predict Type','Histogram Equalization','Edge Detection','Morphological Operations','Spatial Filtering','Frequency Filtering'))

if uploaded_image is not None:
    img = tf.image.decode_image(uploaded_image.read(), channels=3)
    img2 = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2GRAY)
    if choice == 'Predict Type':
        try:
            st.subheader("Predict Type")

            # Display the uploaded image
            st.image(img.numpy(), caption = "Uploaded Image",width = 300)
            
            predicted_class = predict_class(img)
            st.subheader("Result")
            st.info(f"The predicted class for the image is: {predicted_class}")

        except Exception as e:
            st.text(f"An error occurred: {e}")

    elif choice == 'Histogram Equalization':
        st.subheader("Histogram Equalization")
        equalized_image = equalize_image(img2)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img.numpy(),caption = "Original Image",width = 300)
        with col2:
            st.image(equalized_image,caption = "Equalized Image",width = 300)
    
    elif choice == 'Morphological Operations':
        kernel = np.ones((7,7), np.uint8)
        c_im,o_im,e_im,d_im = morph_ops(img2,kernel)
        col1, col2 = st.columns(2)
        with col1:
            st.image(c_im,caption = "Image After Closing",width = 300)
            st.image(e_im,caption = "Image After Erosion",width = 300)
        with col2:
            st.image(o_im,caption = "Image After Opening",width = 300)
            st.image(d_im,caption = "Image After Dilaiton",width = 300)
        
else:
    st.header("About")

    # Introduction
    st.write("This application is designed to help you analyze brain tumor images. It provides various image processing techniques and information about the creator.")

    # Types of Brain Tumors
    st.subheader("Types of Brain Tumors")
    st.write("There are several types of brain tumors, including:")
    st.write("- Glioma Tumor")
    st.write("- No Tumor (Normal Brain)")
    st.write("- Pituitary Tumor")
    st.write("- Meningioma Tumor")

    st.header("Operations this app supports")
    # Edge Detection
    st.subheader("Edge Detection")
    st.write("Edge detection is a technique used to identify the boundaries of objects within an image. It helps in highlighting the edges of structures in brain tumor images, making it easier to analyze them.")

    # Histogram Equalization
    st.subheader("Histogram Equalization")
    st.write("Histogram equalization is a method used to improve the contrast of an image by redistributing the intensity levels. It can enhance the visibility of features in brain tumor images.")

    # Morphological Operations
    st.subheader("Morphological Operations")
    st.write("Morphological operations involve the use of structuring elements to process binary images. These operations, such as dilation and erosion, can help in enhancing or removing certain features in the image.")

    # Creator's Name
    st.header("Creator Info")
    st.write("This Brain Tumor Analysis App was created by : ")
    st.write("- Chris Vinod Kurian")
    st.write("- Drishtti Narwal")
    st.write("- Gaurav Prakash")
    st.write("- Hevardhan Saravanan")

    # Disclaimer
    st.warning("This application is intended for educational and informational purposes. It is not a substitute for professional medical advice. Consult a medical professional for accurate diagnosis and treatment of brain tumors.")

    
