### Handwritten Digit Recognition ###

# Import
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale down data from 0 to 255 so it's more like 0 to 1
# We don't scale the y data since that's the digits (labels)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Define model
# Layers: Input, 2 Hidden, Output
# This is a basic Neural Network (feed forward)
model = tf.keras.models.Sequential()
# First layer (input) which has all of the pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Basically means all of the neurons in the previous layer are connected to this layer
# These are the 2 hidden layers
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) 
# Finally we have the output layer
# Scales the values down so they all add up to 1
# Means you get the percentage/probability of each digit
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) 

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
# epochs is basically how many times is the process going to be repeated
model.fit(x_train, y_train, epochs=3)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)

# Accuracy and Loss
print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')

model.save('digits.keras')

for x in range(0, 10):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img])) # Makes it an array and inverts colours to be correct

    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()