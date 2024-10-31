import numpy as np
import cv2 as cv
import SimpleITK as sitk

def rotate_image_opencv(image, angle_degrees):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle_degrees, 1.0)
    image_rotated = cv.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return image_rotated

def apply_affine_transformation_opencv(image):
    # Define affine transformation parameters
    scale = np.random.uniform(0.9, 1.1)
    shear = np.random.uniform(-10, 10)
    translate_x = np.random.uniform(-0.1, 0.1) * image.shape[1]
    translate_y = np.random.uniform(-0.1, 0.1) * image.shape[0]

    # Define the affine transformation matrix
    matrix = cv.getAffineTransform(
        center=np.array(image.shape[1::-1]) / 2,
        pts1=np.array([[(1 - scale) * image.shape[1], (1 - scale) * image.shape[0]],
                       [(1 + scale) * image.shape[1], (1 - scale) * image.shape[0]],
                       [(1 - scale) * image.shape[1], (1 + scale) * image.shape[0]]]),
        pts2=np.array([[(1 - scale) * image.shape[1] + translate_x, (1 - scale) * image.shape[0] + translate_y],
                       [(1 + scale) * image.shape[1] + translate_x, (1 - scale) * image.shape[0] + translate_y],
                       [(1 - scale) * image.shape[1] + translate_x, (1 + scale) * image.shape[0] + translate_y]]))

    # Apply the affine transformation
    image_transformed = cv.warpAffine(image, matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return image_transformed

def apply_color_jitter_opencv(image):
    # Define color jitter parameters
    brightness_factor = np.random.uniform(0.8, 1.2)
    contrast_factor = np.random.uniform(0.8, 1.2)
    saturation_factor = np.random.uniform(0.8, 1.2)
    hue_factor = np.random.uniform(-0.1, 0.1)

    # Convert the image to HSV color space
    image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Apply color jitter
    image_hsv[..., 2] = image_hsv[..., 2] * brightness_factor
    image_hsv[..., 1] = image_hsv[..., 1] * contrast_factor
    image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)
    image_hsv[..., 0] = (image_hsv[..., 0] + hue_factor * 360) % 360
    image_hsv[..., 1] = image_hsv[..., 1] * saturation_factor
    image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)

    # Convert the image back to RGB color space
    image_jittered = cv.cvtColor(image_hsv, cv.COLOR_HSV2RGB)

    return image_jittered

def translate_image_opencv(image, translation_x, translation_y):
    
    # Define the transformation matrix for translation
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    
    # Apply translation using OpenCV
    translated_image = cv.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    return translated_image

def adjust_brightness_opencv(image, brightness_factor):
    
    # Apply brightness adjustment using OpenCV
    adjusted_image = cv.convertScaleAbs(image, alpha=1, beta=brightness_factor)

    return adjusted_image