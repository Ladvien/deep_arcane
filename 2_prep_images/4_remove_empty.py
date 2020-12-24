#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:33:03 2020

@author: ladvien
"""

import cv2

import sys
import os

import matplotlib.pyplot as plt

image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()



#############
# Parameters
#############

input_path = "/home/ladvien/deep_arcane/images/0_raw/3_all_white/"
output_path = "/home/ladvien/deep_arcane/images/0_raw/4_no_empty/"

threshold = 240

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
    image = iu.remove_empty_images(image, threshold)
        
    try:
        print(f"Writing file {outpout_file_path}")
        cv2.imwrite(outpout_file_path, image)
    except:
        print(f"Removed: {file_path}")
