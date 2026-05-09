import tensrflow as tf
from tensorflow.keras import layers, models 
import os


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()


X_train = X_train / 255.0 # Normalize the pixel values to be between 0 and 1
X_test = X_test / 255.0

model = models.Sequential()
model.add(layers.Conv2D(
    32, # number of filters
    (3,3), # kernel size
    activation='relu', # activation function
    input_shape=(32, 32, 3) # input shape of the images
))


model.add(layer.MaxPooling2D(
    (2, 2) # pool size
))

model.add(layers.Conv2D(
    64, # number of filters
    (3,3), # kernel size
    activation='relu' # activation function
))


model.add(layers.MaxPooling2D(
    (2, 2) # pool size
))


model.add(layers.Conv2D(
    64, # number of filters
    (3,3), # kernel size
    activation='relu' # activation function
))

# add faltten layer to conver the 3D output to 1D
model.add(layers.Flatten())


# add dense layerv(a fully connected neural network)
model.add(layers.Dense(
    64, # number of neurons
    activation='relu' # activation function
))

model.add(layers.Dense(
    10, # number of classes
    activation='softmax' # activation function
))



model.compile(
    optimizer='adam', # optimization algorithm
    loss='sparse_categorical_crossentropy', # loss function
    metrics=['accuracy'] # metrics to evaluate the model
)



model.fit(
    X_train, # training data
    y_train, # training labels
    epochs=10, # number of epochs to train the model
    validation_data=(X_test, y_test) # validation data and labels   
)



os.makedirs('models', exist_ok=True)
model.save("model/cifar10_model.keras")

print("Model Saved")