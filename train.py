# train.py
# Written by Dr. Matthew Smith, Swinburne University of Technology
# Prepared as training material for ADACS Machine Learning workshop
# This is an example of time series (sequence) classification
# for a binary classification problen.

# Import modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import plot_model
from utilities import *

# Create training arrays
# In this demonstration I create our numpy arrays and then
# load each time sequence in one-by-one.
N_train = 200        # Number of elements to train
N_sequence = 128     # Length of each piece of data
N_epochs = 300	     # Number of epochs

# Create the training sequence data (X) and each set's classification (Y).
X_train = np.empty([N_train, N_sequence])
Y_train = np.empty(N_train)

# Load the data from file
for x in range(N_train):
	# This will create x = 0, 1, 2...to N_train-1
	X_train[x,], Y_train[x] = read_training_data(x+1, N_sequence)

# Also create the numpy arrays for the testing data set
N_test = 50
X_test = np.empty([N_test, N_sequence])
Y_test = np.empty(N_test)
for x in range(N_test):
        X_test[x,], Y_test[x] = read_test_data(x+1, N_sequence)

# Create our Keras model - an RNN (in Keras this is a Sequence)
model = Sequential()

# Configure our RNN by adding neural layers with activation functions
model.add(Dense(16, activation='relu',input_dim=N_sequence))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# The commented lines below are designed to offer some insight
# into the use of layers without activation functions.
# The 3 lines above should be commented out before uncommenting these.
# No activation layers
#model.add(Dense(16,input_dim=N_sequence))
#model.add(Dense(1))

# Compile model and print summary
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Fit the model using the training set
history = model.fit(X_train, Y_train, epochs=N_epochs, batch_size=32)

# Plot the history
plot_history(history)

# Final evaluation of the model using the Test Data
print("Evaluating Test Set")
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

