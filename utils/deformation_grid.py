import SimpleITK as sitk
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

## Generate deformation field
#random 3D displacement field in numpy array
# np_displacement_field = np.random.randn(20,256,256,3)

transform = sitk.ReadImage('Z:/pruebas/warp.nii')
transform = transform[:,:,0]
transform = sitk.GetArrayFromImage(transform)


dy, dx = np.gradient(transform)
plt.imshow(transform, cmap='jet')
plt.colorbar(label="Warp", orientation="vertical")
plt.quiver(dx, dy)
plt.show()