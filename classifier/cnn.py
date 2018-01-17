# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:28:45 2017

@author: Aayush
"""

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#random seed for reproducibility
seed = 7
np.random.seed(seed)

#loading dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#normalize inputs from 0-255 to 0.0 - 1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encoding to outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# creating model
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dropout(0.2))
classifier.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
classifier.add(Dropout(0.2))
classifier.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
classifier.add(Dropout(0.2))
classifier.add(Dense(num_classes, activation='softmax'))

# Compiling model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(classifier.summary())


# Fit the model
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Save model with SJSON
# serialize model to JSON
from keras.models import model_from_json
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

#Loading and evaluting the model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\Aayush\\Desktop\\cat_or_dog_1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)

# search for the class
for i in range(0,10):
    if result[0][i] == 1.0:
        index = i
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print (classes[index])

# evaluating the model
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))