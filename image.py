import streamlit as st
import pandas as pd
# import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from PIL import Image

st.image("inno_image.webp")
st.title('Machine Learning for Dog Image Recognition')


model = pickle.load(open("image.pkl",'rb'))

image=st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.image(image)

image = Image.open(image)

arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
img_resized = cv2.resize(arr,(20,20))
img = img_resized.flatten()
if st.button('Submit'):
    st.write(model.predict([img])[0])
