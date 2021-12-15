# Credit for most code here goes to Hrishi Patel
# https://towardsdatascience.com/using-k-means-clustering-for-image-segregation-fd80bea8412d

# Ran into issue with h5py version.
# https://stackoverflow.com/a/65717773
# pip install 'h5py==2.10.0' --force-reinstall

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, shutil

from glob2 import glob

#################
# Parameters
#################
INPUT_FILE_PATH = "/Users/ladvien/Documents/clean_bold_magic_symbols"
OUTPUT_DIR = "/Users/ladvien/deep_arcane/images"

PREVIEW_INPUTS = False
IMG_DIMS = (224, 224)
NUM_PREV_ITEMS = 10

COLOR_CHANNELS = 3

MAX_K = 20
MIN_K = 3

#################
# Prepare
#################
if not os.path.exists(OUTPUT_DIR):
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR)


#################
# Load Images
#################
paths = images = [path for path in glob(INPUT_FILE_PATH + "/*.png")]
images = [
    cv2.resize(cv2.imread(file), (IMG_DIMS[0], IMG_DIMS[1]))
    for file in glob(INPUT_FILE_PATH + "/*.png")
]
images = np.array(np.float32(images).reshape(len(images), -1) / 255)
print(images.shape)


#################
# Extract Features
#################
model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_DIMS[0], IMG_DIMS[1], COLOR_CHANNELS),
)
predictions = model.predict(
    images.reshape(-1, IMG_DIMS[0], IMG_DIMS[1], COLOR_CHANNELS)
)
pred_images = predictions.reshape(images.shape[0], -1)


sil = []
kl = []
for k in range(MIN_K, MAX_K + 1):
    print(f"Calculating K-Means for K = {k}")
    kmeans2 = KMeans(n_clusters = k).fit(pred_images)
    labels = kmeans2.labels_
    sil.append(silhouette_score(pred_images, labels, metric =   "euclidean"))
    kl.append(k)
    
plt.plot(kl, sil)
plt.ylabel("Silhoutte Score")
plt.ylabel("K")
plt.savefig(OUTPUT_DIR + "/silhoutte_scores.png")
plt.show()

# k = 17
# kmodel = KMeans(n_clusters=k, n_jobs=-1, random_state=728)
# kmodel.fit(pred_images)
# kpredictions = kmodel.predict(pred_images)
# shutil.rmtree(OUTPUT_DIR)

# for i in range(k):
#     os.makedirs(f"{OUTPUT_DIR}/cluster_" + str(i))

# for i in range(len(paths)):
#     shutil.copy2(paths[i], f"{OUTPUT_DIR}/cluster_" + str(kpredictions[i]))
