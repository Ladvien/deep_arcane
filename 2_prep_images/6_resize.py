#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 01:56:16 2020

@author: ladvien
"""


import cv2

import numpy as np
import glob, os
import matplotlib.pyplot as plt

#############
# Parameters
#############

input_path = "/home/ladvien/deep_arcane/images/0_raw/5_dilated/"
output_path = "/home/ladvien/deep_arcane/images/0_train_magic_symbol_classifier/"

file_types = ("*.jpg", "*.png")

target_size = (128, 128)

samples = 80

#############
# Extract
#############

if not os.path.exists(output_path):
    os.makedirs(output_path)

os.chdir(input_path)

filenames = []
for file_type in file_types:
    filenames += glob.glob(file_type)


counter = 0
for filename in filenames:
    image_path = input_path + filename
    
    image = cv2.imread(filename, 0)
    
    image = cv2.resize(image, target_size)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    show_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    outpout_file_path = output_path + filename
    print(outpout_file_path)
    
    cv2.imwrite(outpout_file_path, image)
    
    counter += 1
    if counter < samples:
        plt.axis("off")
        plt.imshow(show_image)
        plt.show()
    