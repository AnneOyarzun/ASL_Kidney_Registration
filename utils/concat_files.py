
imgtype = "Native"                                
# imgtype = "Allograft"                               
with open('./data_paths_' + imgtype+ '.txt') as f:    
    studies = f.read().splitlines()

leave_out = "BS01"

for i in range(0, len(studies)): 
    if leave_out in studies[i]: 
        studies[i] = []

for i in range(studies.count([])):
    studies.remove([])


import SimpleITK as sitk
import cv2 as cv
import numpy as np

for i in range(0, len(studies)): 
    image = sitk.GetArrayFromImage(sitk.ReadImage('D:/RM_RENAL/Registration/Pre_Affine/' + imgtype + '/' + studies[i] + 'result.0.nii'))
    image = np.delete(image, 0, 0) # quitamos primera dim
    
    join = sitk.JoinSeriesImageFilter()
    if i == 0:
        joined_image = image
    else:
        #joined_image = join.Execute(joined_image, image)
        joined_image = cv.vconcat([joined_image, image])

print(joined_image.shape)
sitk.WriteImage(sitk.GetImageFromArray(joined_image), 'Z:/pruebas/full_dataset.nii')

np.savez("Z:/pruebas/all_images.npz",joined_image)
# im_v = cv.vconcat([img1, img1])


