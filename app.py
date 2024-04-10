import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import load_model


model = load_model('traffic.h5')

def save_image(img):
    try:
        complete_path = os.path.join("", "image.jpg")
        image = img.resize((50,50))
        image.save(complete_path)
    except Exception as e:
        st.error(f"An error occurred while saving the image: {e}")

def check(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = np.argmax(model.predict(X_test),axis=1)
    return image,Y_pred

def answer(img):
    plot, prediction = check(img)
    s = [str(i) for i in prediction]
    a = "".join(s)
    a = int(a)
    return a

classes = ['Speed limit (20km/h)',
            'Speed limit (30km/h)',
            'Speed limit (50km/h)',
            'Speed limit (60km/h)',
            'Speed limit (70km/h)',
            'Speed limit (80km/h)',
            'End of speed limit (80km/h)',
            'Speed limit (100km/h)',
            'Speed limit (120km/h)',
            'No passing',
            'No passing veh over 3.5 tons',
            'Right-of-way at intersection',
            'Priority road',
            'Yield',
            'Stop',
            'No vehicles',
            'Veh > 3.5 tons prohibited',
            'No entry',
            'General caution',
            'Dangerous curve left',
            'Dangerous curve right',
            'Double curve',
            'Bumpy road',
            'Slippery road',
            'Road narrows on the right',
            'Road work',
            'Traffic signals',
            'Pedestrians',
            'Children crossing',
            'Bicycles crossing',
            'Beware of ice/snow',
            'Wild animals crossing',
            'End speed + passing limits',
            'Turn right ahead',
            'Turn left ahead',
            'Ahead only',
            'Go straight or right',
            'Go straight or left',
            'Keep right',
            'Keep left',
            'Roundabout mandatory',
            'End of no passing',
            'End no passing veh > 3.5 tons' ]



st.markdown("<h1 style='text-align: center;'>TrafficGuardian</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your Own Traffic Signal Detector</h3>", unsafe_allow_html=True)

uploaded_file=st.file_uploader("Please Enter Your Sign Image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    save_image(image)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        new_img=image.resize((600, 400))
        st.image(new_img)
    with col3:
        st.write(' ')
    centered_style = """
        <style>
        .centered {
            text-align: center;
        }
        </style>
    """
    st.markdown("<h3 style='text-align: center;'>Predicted traffic sign is:</h3>", unsafe_allow_html=True)
    img='image.jpg'
    a=answer(img)
    ans=classes[a]
    print(ans)
    st.markdown(f"<h3 style='text-align: center;'>{ans}</h3>", unsafe_allow_html=True)
