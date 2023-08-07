import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

st.set_page_config(
    page_title='Predict',
    layout='wide',
    initial_sidebar_state='expanded'
)

# load model
best_model = load_model('model_vgg.h5')

def img_predict(img):
    img_array = np.array(img)
    img_resized = tf.image.resize(img_array, size=(255, 255)) / 255.0
    pred = best_model.predict(np.expand_dims(img_resized, axis=0))
    res = np.argmax(pred)
    if res == 0:
        return "adidas"
    elif res == 1:
        return "converse"
    else:
        return "nike"

def run():
    # variable image
    img = None

    st.title("Image Classification of Shoes")

    # Image Upload Option
    choose = st.selectbox("Select Input Method", ["Upload an Image", "URL from Web"])

    if choose == "Upload an Image":  # If user chooses to upload an image
        file = st.file_uploader("Upload an image...", type=["jpg", "png", 'Tiff'])
        if file is not None:
            img = Image.open(file)
    else:  # If user chooses to upload an image from URL
        url = st.text_area("URL", placeholder="Put URL here")
        if url:
            try:  # Try to get the image from the URL
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            except:  # If the URL is not valid, show error message
                st.error("Failed to load the image. Please use a different URL or upload an image.")

    if img is not None:
        predict = st.button("Predict")
        if predict:
            prediction = img_predict(img)
            st.write(f"Predicted shoe brand: {prediction}")
            st.image(img, use_column_width=True)

if __name__ == '__main__':
    run()
