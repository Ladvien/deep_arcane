#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:31:46 2020

@author: ladvien
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:33:03 2020

@author: ladvien
"""

import cv2

import numpy as np
import glob, os
import matplotlib.pyplot as plt

#############
# Parameters
#############

input_path = "/home/ladvien/deep_arcane/images/0_raw/4_no_empty/"
output_path = "/home/ladvien/deep_arcane/images/0_raw/5_dilated/"

file_types = ("*.jpg", "*.png")

threshold = 150

kernel = np.ones((2,2), np.uint8)

samples = 10

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
    
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(image, axis=0)
    
    avg_color = avg_color.mean()
    
    # Invert the image if the average darkness is below
    # threshold.
    if avg_color < threshold:
        if counter < samples:
            plt.axis("off")
            plt.imshow(image)
            plt.show()
   
        image = ~image
        image = cv2.dilate(image, kernel, iterations =  1)
        image = ~image
        
        if counter < samples:
            plt.axis("off")
            plt.imshow(image)
            plt.show()

    outpout_file_path = output_path + filename
    print(outpout_file_path)
    cv2.imwrite(outpout_file_path, image)
    
    counter += 1

    