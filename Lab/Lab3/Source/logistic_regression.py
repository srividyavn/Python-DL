from __future__ import print_function
import keras
from keras_preprocessing import sequence
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier

dataset = pd.read_csv('heart.csv',index_col=0)
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X = (X - X.mean()) / (X.max() - X.min())
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

np.random.seed(155)
def createmodel():
	model = Sequential() # create model
	model.add(Dense(40, input_dim=12, activation='relu')) # hidden layer
	model.add(Dense(20, input_dim=40, activation='relu'))
	model.add(Dense(1, activation='sigmoid')) # output layer

	from tensorboardcolab import *
	tbc = TensorBoardColab()
    #tbCallBack= keras.callbacks.TensorBoard(log_dir='./log1', write_images=True)

	model.compile(loss= keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.adamax(),
                  metrics=['accuracy'])
	return model



#fit the model
history = model.fit(X_train, Y_train,batch_size=256,epochs=30,verbose=1,
           validation_data=(X_test, Y_test), callbacks=[TensorBoardColabCallback(tbc)])

from keras import backend as K
model = KerasClassifier(build_fn=createmodel,verbose=0)
batch_size= [32, 256,1024]
epochs = [10, 20,30]
param_grid= dict(epochs=epochs,batch_size = batch_size)
from sklearn.model_selection import GridSearchCV
grid  = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result= grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))