import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import tensorflow as tf
# import the necessary packages for image recognition
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras import backend as K
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib
from classify_new import classify_covid, classify_tb, classify_xray
plt.style.use("ggplot")

# set page layout
st.set_page_config(
    page_title="Image Classification App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DetX-Ray")

page = st.sidebar.radio("Navigation",["Home", "Register", "Covid & Pneumonia Classifier"])
logged_in = False

if os.path.exists("user_database.py"):
    user_dict = np.read("user_database.np")
else:
    user_dict = defaultdict(list)


if page=="Home":
    # st.warning("If you are visiting for the first time, please consider registeing yourself. You can register yourself from choosing the appropriate action on the left sidebar")
    # st.markdown(""" ## About us <br>
    # CogXRLabs is a For-profit Social Enterprise with an aim to commence Healthcare 4.0 in India. It is a confluence of state-of-the-art cutting-edge technologies including, but not limited to, AI (Machine Learning and Deep Learning), Extended Reality (Virtual, Augmented and Mixed), and Brain Computer Interfaces. <br>
    # ## Vision <br>
    # Our vision is to introduce improved, transformational suite of products and expert services that address the current challenges faced by the healthcare industry in India. The aim is to commence the era of Healthcare 4.0 by providing patients with better, more value-added, and more cost-effective healthcare services while also improving the industry's efficacy and efficiency.
    # """,True)
    st.write("DetX-Ray is a web application to provide primal checkup of your X-Ray if it has Covid, Pneumonia, or just Normal")
    st.write("Upload the X-Ray image and get your results within seconds!")
    
    
elif page=="Register":
    #st.set_page_config("Register", layout='centered',page_icon = ":clipboard:")

    st.title("Registration Form")



    first_name, last_name = st.columns(2)
    first_name.text_input("First Name")
    last_name.text_input("Last Name")
    
    age, gender, = st.columns([3,1])
    age.number_input("Age (in Years)",min_value = 1,max_value=100,value = 30,step = 1)
    gender.selectbox("Gender",["Male","Female","Others"],index = 0)

    email_id, mobile_number = st.columns([2,2])
    email_id.text_input("Email ID")
    mobile_number.text_input("Mobile Number")

    col5 ,col6 ,col7  = st.columns(3)
    username = col5.text_input("Username")
    password =col6.text_input("Password", type = "password")
    col7.text_input("Repeat Password" , type = "password")

    but1,but2,but3 = st.columns([1,5,1])

    agree  = but1.checkbox("I Agree")

    if but3.button("Submit"):
        if agree: 
            user_dict["first_name"].append(first_name)
            user_dict["last_name"].append(last_name)
            user_dict["age"].append(age)
            user_dict["gender"].append(gender)
            user_dict["username"].append(username)
            user_dict["password"].append(password)
            user_dict["email_id"].append(email_id)
            user_dict["mobile_number"].append(mobile_number)
            st.success("Done")
        else:
            st.warning("Please Check the T&C box")

elif page=="Covid & Pneumonia Classifier":
    image = st.file_uploader(f"Choose an X-Ray Image", ['jpg', 'jpeg','png'])

    if image:
        st.image(image)
        img = Image.open(image).convert('RGB') # 3 channels
        st.write("")
        st.write("Classifying now...")
        label = classify_covid(img, 'Covid-DenseNet121.h5')
        if label == 0:
            st.header("X-ray image has Normal")
        elif label==1:
            st.header("X-Ray scan is Pneumonia")
        else:
            st.header("X-Ray has Covid")

        # model1=tf.keras.models.load_model('Covid-DenseNet121.h5')
        # img_test = np.array(img.resize((224,224)))
        # img_test = np.expand_dims(img_test, axis=0)
        # st.write(img_test.shape)
        # preds = model1.predict(img_test)
        # class_idx = np.argmax(preds)
        # # class_output = model1.output[:, class_idx]
        # # last_conv_layer = model1.get_layer('conv5_block16_concat')

        # st.write(class_idx)
        # grads = K.gradients(class_output, last_conv_layer.output)[0]
        # # grads = K.gradients(class_output, last_conv_layer.output)[0]
        # pooled_grads = K.mean(grads, axis=(0, 1, 2))
        # iterate = K.function([model1.input], [pooled_grads, last_conv_layer.output[0]])
        # pooled_grads_value, conv_layer_output_value = iterate([x])
        # for i in range(1024):
        #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        # # i=np.where(preds>0.5,1,0)
        # # i.flatten()
        
        # heatmap = np.mean(conv_layer_output_value, axis=-1)
        # heatmap = np.maximum(heatmap, 0)
        # heatmap /= np.max(heatmap)
        # st.write(heatmap.shape)
