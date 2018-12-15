# view.py
# Written by Dr. Matthew Smith, Swinburne University of Technology
# Load a single data set and plot using matplotlib
# This assumes you have X11 forwarded if you are running remotely.
# Mac O/S users will need XQuartz installed and running.
# Usage: python view.py 2  <enter>
# This will load the time sequence data from ID=2 in the training set.


import sys
import numpy as np
from utilities import *

# Parse the arguments
no_arg = len(sys.argv)
if (no_arg == 2):
	plot_ID = int(sys.argv[1])
else:
	print("Usage: python view.py <Data_ID>")
	print("where Data_ID is a number.")
	print("Example: python view.py 2")
	sys.exit()

# Load training data
N_sequence = 128;    # Length of each piece of data

# Load a specific data file and plot the results
X_train, Y_train = read_training_data(plot_ID,N_sequence)
plot_results(plot_ID,X_train)

