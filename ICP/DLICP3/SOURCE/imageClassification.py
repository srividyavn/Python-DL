# Simple CNN model for CIFAR-10
import numpy
#from keras.datasets import cifar10
from keras.datasets import mnist
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
from keras.models import load_model
import numpy as np
from tensorflow import keras

K.set_image_dim_ordering('th')


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# input image dimensions
img_rows, img_cols = 28, 28


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(1,28,28), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph', write_images=True)

model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=2, batch_size=32,  callbacks=[tbCallBack])

# Fit the model
#model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=2, batch_size=32)

model.save('fashionmnist_update.h5')


#loading model
model = load_model('fashionmnist_update.h5')

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



print(x_test.shape)
#Predictions for first four images

for i in range(0,4):
    predicted_value = model.predict(x_test[[i],:])
    predict_classes = model.predict_classes(x_test[[i],:])
    actual_value = y_test[[i],:]
    print("Actual Value for :" + str(i) + 'st Image' + str(np.argmax(actual_value)))
    print("Predicted Value for " + str(i) + 'st Image' + str(predict_classes))