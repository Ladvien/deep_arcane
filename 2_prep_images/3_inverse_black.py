#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:10:36 2020

@author: ladvien
"""


import cv2

import os
import sys

import matplotlib.pyplot as plt

image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()



#############
# Parameters
#############

input_path = "/home/ladvien/deep_arcane/images/0_raw/2_black_and_white/"
output_path = "/home/ladvien/deep_arcane/images/0_raw/3_all_white/"

threshold = 80

samples = 10

#############
# Extract
#############

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_paths = iu.get_image_files_recursively(input_path)

counter = 0
for file_path in file_paths:
    file_name = file_path.split("/")[-1]
    outpout_file_path = output_path + file_name
    
    image = cv2.imread(file_path)
    image = iu.invert_mostly_black_images(image, threshold)
        
    print(f"Writing file {outpout_file_path}")
    cv2.imwrite(outpout_file_path, image)
    
    counter += 1
    if counter < samples:
        plt.axis("off")
        plt.imshow(image)
        plt.show()