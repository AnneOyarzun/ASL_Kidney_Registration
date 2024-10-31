from array import ArrayType
import matplotlib.pyplot as plt
import numpy as np



def visualize_arr(arr): 
    plt.figure()
    plt.imshow(arr, cmap='gray')
    plt.show()


def overlay_mask(image, mask):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Overlay the mask
    ax.imshow(mask, alpha=0.4, cmap='Reds')

    # Show the plot
    plt.show()



