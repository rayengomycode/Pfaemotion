import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
img_height = 48
img_width = 48

model = tf.keras.models.load_model("EmotionDetecttor_2")
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
st.header("Welcome Emotion Detection App")
select = st.selectbox("Choose an option", ["Upload a picture", "Use your Webcam"])
if select == "Upload a picture":
    img_file_buffer = st.file_uploader("Choose a file")
    if img_file_buffer is not None:
    # To read file as bytes:
        st.image(img_file_buffer)
        img = Image.open(img_file_buffer)
        
        img = img.convert('RGB')

    
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        scoring = {}
        for i in range(len(class_names)):
            scoring[class_names[i]] = score[i].numpy()
        df = pd.DataFrame(scoring, index=[0])
        for col in df.columns:
            df[col]= df[col].map('{:,.2f}'.format)
        st.dataframe(df.style.highlight_max(axis=1))
        st.markdown(
            "This image most likely belongs to **_{}_** with a **_{:.2f}_** percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
if select =="Use your Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer:
        st.image(img_file_buffer)
        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)
        img = img.convert("L")
        img = img.convert('RGB')
        img_array = tf.keras.utils.img_to_array(img.resize((48,48)))
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        scoring = {}
        for i in range(len(class_names)):
            scoring[class_names[i]] = score[i].numpy()
        df = pd.DataFrame(scoring, index=[0])
        for col in df.columns:
            df[col]= df[col].map('{:,.2f}'.format)
        st.dataframe(df.style.highlight_max(axis=1))
        st.markdown(
            "This image most likely belongs to **_{}_** with a **_{:.2f}_** percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )