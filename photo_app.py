import numpy as np
import streamlit as st
import cv2
from PIL import Image
import mediapipe as mp
import  face_recognition

mp_drawing=mp.solutions.drawing_utils
mp_selfie_segmentation=mp.solutions.selfie_segmentation
model=mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
st.title("Photo changing app")
st.subheader("Create different shades of your photo")
st.write("This app is about creating different shade of life in a photo")

add_selectbox =st.sidebar.selectbox("What Operation you like to perform??",("About","Grayscale","Shade","Blending","Background change","Face Matching"))

if add_selectbox=="About":
    st.write("This App will help you convert the color of your photo.")
    st.write("Can change background of your photo.")
    st.write("Blend it with another photo.")

elif add_selectbox=="Grayscale":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image=np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        st.image(gray_image)

elif add_selectbox=="Shade":
    image_file_path= st.sidebar.file_uploader("Upload image")
    add_radio=st.sidebar.radio("Choose shade",("Red","Blue","Green"))
    if image_file_path is not None:
        image= np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeroes=np.zeros(image.shape[:2],"uint8")
        b,g,r=cv2.split(image)
        if add_radio=="Red":
            red_image=cv2.merge([r,zeroes,zeroes])
            st.image(red_image)
        elif add_radio=="Blue":
            blue_image=cv2.merge([zeroes,zeroes,b])
            st.image(blue_image)
        elif add_radio=="Green":
            green_image=cv2.merge([zeroes,g,zeroes])
            st.image(green_image)

elif add_selectbox=="Blending":
    image_file_path=st.sidebar.file_uploader("Upload image")
    image_file_path2=st.sidebar.file_uploader("Upload image to blend with")
    if image_file_path and image_file_path2 is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        image2 = np.array(Image.open(image_file_path2))
        st.sidebar.image(image2)
        image2=cv2.resize(image2,(image.shape[1],image.shape[0]))
        alpha=st.slider("select visibility for main image",0.0,1.0,0.8)
        beta=st.slider("select visibility for another image",0.0,1.0,0.3)
        gamma=st.slider("select Gamma",0.0,1.0,0.1)
        blended_image=cv2.addWeighted(image,alpha,image2,beta,gamma)
        st.image(blended_image)

elif add_selectbox=="Background change":
    image_file_path=st.sidebar.file_uploader("Upload image")
    background_image=st.sidebar.file_uploader("Upload Background image")
    if image_file_path is not None:
        image=np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model.process(image)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        if background_image is not None:
            bg_image = np.array(Image.open(background_image))
            bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            output_image = np.where(condition, image, bg_image)
            st.image(output_image)
        else:
            background_image=np.zeros(image.shape,"uint8")
            background_image[:]=(0,255,0)
            output_image = np.where(condition, image, background_image)
            st.image(output_image)

elif add_selectbox=="Face Matching":
    image_file_path=st.sidebar.file_uploader("Upload Real Image")
    image_file_path2=st.sidebar.file_uploader("Upload image to match")
    if image_file_path and image_file_path2 is not None:
        image_train=np.array(Image.open(image_file_path))
        image_test=np.array(Image.open(image_file_path2))
        image_encodings_train = face_recognition.face_encodings(image_train)[0]
        image_location_train = face_recognition.face_locations(image_train)[0]
        image_encodings_test = face_recognition.face_encodings(image_test)[0]

        results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)[0]
        dst = face_recognition.face_distance([image_encodings_train], image_encodings_test)
        if results:
            st.write(results)
            cv2.rectangle(image_train, (image_location_train[3], image_location_train[0]),
                          (image_location_train[1], image_location_train[2]), (0, 255, 0), 2)
            st.image(image_train)
        else:
            st.write(results)
            st.write("Could not recognize face")


