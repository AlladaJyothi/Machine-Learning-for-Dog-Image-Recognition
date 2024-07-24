import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
from PIL import Image

st.image(r"C:\Users\DELL\Pictures\inno_image.webp")
name=st.title(r'Machine Learning for Dog Image Recognition')


model = pickle.load(open(r'C:\\Users\\DELL\\Machine Learning\\image.pkl','rb'))

image=st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.image(image,caption='Uploaded Image',use_column_width=True)

image = Image.open(image)

arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
img_resized = cv2.resize(arr,(20,20))
img = img_resized.flatten()
if st.button('Submit'):
    st.write(model.predict([img])[0])
