import numpy as np
import mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
#normalize pixel values from 0-1
train_images = train_images/255
test_images = test_images/255
#flatten 28x28 images to 784
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
#build the model with 2 hidden layers of 128 and output layer of 10
model = keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=784))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train model
model.fit(train_images, utils.to_categorical(train_labels), epochs=10, batch_size=32)
#test model
test = model.predict(test_images[:10])
print('Predicted: ' + str(np.argmax(test, axis=1)))
print('Actual: ' + str(test_labels[:10]))
for i in range(10):
    image = test_images[i]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    if(np.argmax(test[i])==test_labels[i]): print("Prediction: " + str(np.argmax(test[i])) + ' correct!')
    else: print("Prediction: " + str(np.argmax(test[i])) + ' incorrect! Correct Answer: ' + str(test_labels[i]))
    plt.show()