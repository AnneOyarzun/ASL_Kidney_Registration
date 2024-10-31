import SimpleITK as sitk
import numpy as np

u = sitk.ReadImage('Z:/SyntheticASLData/SyntheticASLData/Model77/PCASL/Healthy/Slice_1.nii')
print(u.GetSize())

squeezed = np.squeeze(sitk.GetArrayFromImage(u))
print(squeezed.shape)
my_img = np.zeros([3, 96, 96])
main_path = 'Z:/SyntheticASLData/SyntheticASLData/Model77/PCASL/Healthy/prueba/'
for i in range(0,len(squeezed)): 
    save_path = main_path + str(i+1) + '.nii'
    for s in range(0,3):
        my_img[s,:,:] = squeezed[i, :, :]
        sitk.WriteImage(sitk.GetImageFromArray(my_img), save_path)
