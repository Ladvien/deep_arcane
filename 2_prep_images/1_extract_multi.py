#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 00:14:23 2020

@author: ladvien

    This script finds subimages in an image, extracts them, and saves
    them to a file.

"""

import os
import sys

image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()

#############
# Parameters
#############

input_path = "/home/ladvien/deep_arcane/images/0_raw/0_scraped"
output_path = "/home/ladvien/deep_arcane/images/0_raw/1_extracted"

minimum_size = 30

dry_run = False

#############
# Extract
#############

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_paths = iu.get_image_files_recursively(input_path)

index = 0
for file_path in file_paths:
    
    filename = file_path.split("/")[-1].split(".")[0]
    
    iu.save_subimages(filename, file_path, output_path, minimum_size = minimum_size)
    index += 1
    if index > 10 and dry_run:
        break
    
