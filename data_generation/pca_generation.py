from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import random 
import numpy as np
import matplotlib.pyplot as plt

def template_pca(img_series, label=None): 
    ## Apply the PCA  
    # ---------------------------------------------
    pca = PCA(0.95)
    indexes = list(range(0, img_series.shape[1],1))
    random.shuffle(indexes)
    pca.fit_transform(img_series)
    weights = pca.components_

    weighted_img = np.zeros([int(np.sqrt(img_series.shape[0])), int(np.sqrt(img_series.shape[0]))])
    
    for n_weights in range(0, len(weights[0])): 
        resh_img = np.reshape(img_series[:, indexes[n_weights]], (int(np.sqrt(img_series.shape[0])), int(np.sqrt(img_series.shape[0]))))
        if n_weights == 0:
            weighted_img = np.multiply(resh_img, (weights[0][n_weights]/sum(weights[0])))
            # plt.imshow(weighted_img, cmap = "gray")
        else:
            weighted_img = weighted_img + np.multiply(resh_img, (weights[0][n_weights]/sum(weights[0])))
            # plt.imshow(weighted_img, cmap = "gray")

        # if label is None:
        #     return weighted_img
        # else:
        #     weighted_img[weighted_img>0.5] = 1
        #     weighted_img[weighted_img<0.5] = 0
        #     return weighted_img
    
    return weighted_img
    

    


# def template_pca(img_series):
#     # img_series is of shape (9216, 17), where 9216 = 96*96 and 17 is the number of images.
    
#     # Apply PCA to the transposed data
#     pca = PCA(n_components=1)  # We only need the first principal component
#     pca.fit(img_series.T)
    
#     # Get the weights from the first principal component
#     weights = pca.components_[0]
    
#     # Reconstruct the template image using these weights
#     weighted_img = np.zeros(9216)
    
#     for i in range(img_series.shape[1]):  # Iterate over the number of images
#         weighted_img += weights[i] * img_series[:, i]
    
#     # Reshape the weighted image back to the original shape (96x96)
#     weighted_img = weighted_img.reshape((96, 96))
    
#     # Normalize the image to be between 0 and 1 for visualization
#     weighted_img = (weighted_img - np.min(weighted_img)) / (np.max(weighted_img) - np.min(weighted_img))
    
#     return weighted_img

    
