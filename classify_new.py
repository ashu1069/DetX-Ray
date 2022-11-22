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

# # initialize the input image shape (224x224 pixels) along with
# # the pre-processing function (this might need to be changed
# # based on which model we use to classify our image)
# inputShape = (224, 224)
# preprocess = imagenet_utils.preprocess_input
# # if we are using the InceptionV3 or Xception networks, then we
# # need to set the input shape to (299x299) [rather than (224x224)]
# # and use a different image pre-processing function
# if network in ("Inception", "Xception"):
#     inputShape = (299, 299)
#     preprocess = preprocess_input

# Network = MODELS[network]
# model = Network(weights="imagenet")

# # load the input image using PIL image utilities while ensuring
# # the image is resized to `inputShape`, the required input dimensions
# # for the ImageNet pre-trained network
# image = Image.open(BytesIO(bytes_data))
# image = image.convert("RGB")
# image = image.resize(inputShape)
# image = img_to_array(image)
# # our input image is now represented as a NumPy array of shape
# # (inputShape[0], inputShape[1], 3) however we need to expand the
# # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# # so we can pass it through the network
# image = np.expand_dims(image, axis=0)
# # pre-process the image using the appropriate function based on the
# # model that has been loaded (i.e., mean subtraction, scaling, etc.)
# image = preprocess(image)

# preds = model.predict(image)
# predictions = imagenet_utils.decode_predictions(preds)
# imagenetID, label, prob = predictions[0][0]