import cv2 as cv
import numpy as np

def compute_mean(img, mask):
    # Threshold the mask (convert values to 0 or 255)
    _, mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    masked_image = cv.bitwise_and(img, img, mask = mask)
    # Convert masked_image to float type to handle NaNs
    masked_image = masked_image.astype(np.float32)
  # Set masked pixels with value 0 to NaN
    masked_image[masked_image == 0] = np.nan

    return np.nanmean(masked_image.reshape(-1))

def compute_tsnr(mean_values): 
    tsnr_value = np.nanmean(mean_values)/np.nanstd(mean_values)
    
    return tsnr_value

