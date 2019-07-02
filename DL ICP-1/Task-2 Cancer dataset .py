import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd

from sklearn.model_selection import train_test_split
dataset = pd.read_csv("./datasets/Breas Cancer.csv")

x1=dataset.drop(['diagnosis', 'id', 'fractal_dimension_worst'],axis=1)
y1=dataset['diagnosis']

y1 = y1.map({'B': 1, 'M':1})

import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(x1, y1, test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(50, input_dim=30, activation='sigmoid')) # hidden layer
my_first_nn.add(Dense(25, activation='sigmoid')) # hidden layer
my_first_nn.add(Dense(2, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test,verbose=0))

"""
Accuracy is increasing to a certain point when we add layers. 
If we add more number of layers then slowly the accuracy is decreasing
If random seed is given then -> the accuracy doesnot change as seed takes care of setting same rows for every iteration
"""