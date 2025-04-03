import os
import json
from PIL import Image


import numpy as np
import tensorflow as tf
import streamlit as st

FILE_ID = "15YrUk8gQU2TAsWB3qxXRi8kNdqU28cHU"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
print(f"tensor flow version:{tf.__version__}")
working_dir=os.path.dirname(os.path.abspath(__file__))
model_path=f"{working_dir}/plant_disease_prediction_model.h5"

if not os.path.exists(model_path):
    st.write("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, model_path, quiet=False)
print("Model Path:", os.path.abspath(model_path))  # Debugging line
model=tf.keras.models.load_model(model_path)


class_indices=json.load(open(f'{working_dir}/class_indices.json'))
def load_and_preprocess_images(img_path,target_size=(224,224)):
  img=Image.open(img_path)
  img=img.resize(target_size)
  img_arr=np.array(img)
  img_arr=np.expand_dims(img_arr,axis=0)
  img_arr=img_arr.astype('float32')/255.
  return img_arr,img

def predict_image_class(model,img_path,class_indices):
  
  predictions=model.predict(img_path)
  predicted_class_index=np.argmax(predictions,axis=1)[0]

  if str(predicted_class_index) not in class_indices:
        print("Warning: Predicted class index not found in class_indices.json")
        return "Unknown"
  
  predicted_class_name=class_indices[str(predicted_class_index)]
  return predicted_class_name

st.title('Plant Disease Classifier')

uploaded_image=st.file_uploader("Upload an image",type=['jpg','jpeg','png'])

if uploaded_image is not None:
  img1,img2=load_and_preprocess_images(uploaded_image)
  col1,col2=st.columns(2)

  with col1:
    resized_img=img2.resize((150,150))
    st.image(resized_img)
  with col2:
    if(st.button('Classify')):
      prediction=predict_image_class(model,img1,class_indices)

      if prediction:
        st.success(f'Predicted class :{str(prediction)} ')  
      else:
        st.warning(f'predicted class : {uploaded_image.name}')
