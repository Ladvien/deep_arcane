import cv2

import numpy as np

import os
import sys
from glob import glob

import matplotlib.pyplot as plt

class ImageUtils:

    def __init__(self, ):
        pass


    def find_subimage_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        return cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    def find_subimages(self, image, minimum_size, verbose = 0):

        images = []

        # Find contours
        cnts = self.find_subimage_contours(image)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            
            # # Skip if too small.
            if w < minimum_size or h < minimum_size:
                continue
            
            if verbose > 0:
                print(f"Found image -- W: {w} H: {h}")

            images.append(image[y : y + h, x : x + w])

        return images

    def save_subimages(self, filename, image_path, output_path, minimum_size, verbose = 0):
        
        image = cv2.imread(image_path, minimum_size)
        images = self.find_subimages(image, minimum_size, verbose)

        for i, image in enumerate(images):
            write_path = f"{output_path}/{filename}_{i}.png"
            print(f"Saving: {write_path}")
            cv2.imwrite(write_path, image)


    def convert_image_to_bw(self, image, threshold):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 255, threshold, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)            
        thresh, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image

    def midpoint(img):
        maxf = maximum_filter(img, (3, 3))
        minf = minimum_filter(img, (3, 3))
        return (maxf + minf) / 2

    def contraharmonic_mean(img, size, Q):
        num = np.power(img, Q + 1)
        denom = np.power(img, Q)
        kernel = np.full(size, 1.0)
        return cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)

    def get_image_files_recursively(self, root_dir, exclude_files = []):
        file_types = ("*.jpg", "*.jpeg", "*.png")
        files = []

        for file_type in file_types:
            for dir, _, _ in os.walk(root_dir):
                print(dir)
                files.extend(glob(os.path.join(dir, file_type))) 

        files = [file for file in files if file not in exclude_files]

        return files

    def invert_mostly_black_images(self, image, threshold):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(image, axis=0)

        avg_color = avg_color.mean()

        # Invert the image if the average darkness is below
        # threshold.
        if avg_color < threshold:
            image = ~image
        return image


    def remove_empty_images(self, image, threshold):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(image, axis=0)
        
        avg_color = avg_color.mean()
        
        # Invert the image if the average darkness is below
        # threshold.
        if avg_color < threshold:
            return image

        return None