# Most credits for this code goes to Siddharth Yadav
# https://www.kaggle.com/thebrownviking20/clustering-images-w-neural-network-bayesian-opt
import os

# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
import concurrent.futures
import time
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

from glob import glob

from PIL import Image
import numpy as np
import pandas as pd

#################
# Parameters
#################
INPUT_FILE_PATH = "/Users/ladvien/Documents/clean_bold_magic_symbols"

PREVIEW_INPUTS = False
IMG_DIMS = (128, 128)
NUM_PREV_ITEMS = 10


#################
# Load Images
#################
imgs = []
for file_path in glob(INPUT_FILE_PATH + "/*.png"):
    img = Image.open(file_path).convert("1")  # Convert to BW on load.
    imgs.append(np.asarray(img).flatten())

som_data = pd.DataFrame(imgs)
#################
# Preview
#################
if PREVIEW_INPUTS:
    f, ax = plt.subplots(1, NUM_PREV_ITEMS)
    f.set_size_inches(IMG_DIMS[0] + 5, IMG_DIMS[1] + 5)
    for i in range(NUM_PREV_ITEMS):
        ax[i].imshow(som_data[i].reshape(IMG_DIMS[0], IMG_DIMS[1]))
    plt.show()


#################
# Preview
#################
# Initializing the map
start_time = time.time()

# The map will have x*y = 50*50 = 2500 features
som = MiniSom(x=50, y=50, input_len=som_data.shape[0], sigma=0.5, learning_rate=0.4)

# There are two ways to train this data
# train_batch: Data is trained in batches
# train_random: Random samples of data are trained. Following line of code provides random weights as we are going to use train_random for training
som.random_weights_init(som_data)
