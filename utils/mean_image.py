import numpy as np
import cv2 as cv

def calculate_average_img(img_series): 
    mean_image = np.mean(img_series, axis=0)  # Forma: (96, 96)
    return mean_image
