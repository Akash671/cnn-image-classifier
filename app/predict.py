import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("model/cifar10_model.keras")

class_names = ['airplane', 
               'automobile', 
               'bird', 
               'cat', 
               'deer',
               'dog', 
               'frog', 
               'horse',
               'ship', 
               'truck']




def predict(image):
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return predicted_class, confidence


