import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras import backend as K

def classify_covid(img, model):
    # Load the model
    model = tf.keras.models.load_model(model)
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    # #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability


def classify_xray(img, model):
    #load model
    model = tf.keras.models.load_model(model)
    #create the array of the right shape to feed into the model
    # image = img
    # size = (224,224,3)
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = img.resize((224,224))
    #convert image to numpy array
    img_arr = np.array(image)
    #print(img_arr.shape)
    img_arr = np.expand_dims(img_arr, axis=0)
    normalized_img_arr = (img_arr.astype(np.float32)/255.0)
    # data = np.array(img.resize(224,224))
    # data = np.expand_dims(data, axis=0)
    

    preds = model.predict(normalized_img_arr)
    # count = 0
    # for i in range(0,6,1):
    #     if preds[0][i]>=0.5:
    #         count+=1
    #     else:
    #         count+=0
    return preds




def classify_tb(img, model):

    model = tf.keras.models.load_model(model)
    #create the array of the right shape to feed into the model
    data = np.ndarray(shape=(1,320, 320, 3), dtype=np.float32)
    image = img
    size = (320,320)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #convert image to numpy array
    img_arr = np.asarray(image)
    normalized_img_arr = (img_arr.astype(np.float32)/127.0)-1

    data[0] = normalized_img_arr

    preds = model.predict(data)
    return preds
