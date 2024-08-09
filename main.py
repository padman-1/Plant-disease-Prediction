from fastapi import FastAPI, File, UploadFile

import uvicorn

import numpy as np

from io import BytesIO
from PIL import Image
import tensorflow as tf 
from os import environ

app = FastAPI()
MODEL_POTATO = tf.keras.models.load_model("./saved_models/potato/potatoes.keras", compile=False)
# MODEL_POTATO = tf.keras.layers.TFSMLayer("./saved_models/1",  compile=False)
# MODEL_CORN = tf.keras.models.load_model("./saved_models/corn/1")
# MODEL_PEPPER = tf.keras.models.load_model("./saved_models/pepper/2")


CLASS_NAMES_POTATO = ["Early Blight", "Late Blight" ,"Healthy"]
CLASS_NAMES_CORN = ["Corn Blight","Corn Common Rust","Corn Gray Leaf Spot","Healthy"]
CLASS_NAMES_PEPPER = ["Bell Bacterial Spot","Healthy"]

# def read_file_as_image(data) -> np.array:
#     image = np.array(Image.open(BytesIO(data )))
#     return image
    
def read_file_as_image(data) -> np.array:
    # image = np.array(Image.open(BytesIO(data )))
    image = Image.open(BytesIO(data ))
    image = image.resize((256, 256))
    return np.array(image)    
    
@app.post("/potato")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL_POTATO.predict(img_batch)
    
    predicted_class = CLASS_NAMES_POTATO [np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predictions)
    return  {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    
    
    
# @app.post("/corn")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
    
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = MODEL_CORN.predict(img_batch)
    
#     predicted_class = CLASS_NAMES_CORN [np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return  {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
  
    
# @app.post("/pepper")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
    
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = MODEL_PEPPER.predict(img_batch)
    
#     predicted_class = CLASS_NAMES_PEPPER[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return  {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
    
    







if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=environ.get("PORT",8000))
