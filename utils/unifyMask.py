import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# mask = sitk.ReadImage('G:/.shortcut-targets-by-id/1L4UF0b3NvLVzjuyg84AcKuEyF-HRltzn/RM-RENAL_CUN&UPNA/DATA/Native/V08/mask/seg.nii')
mask_left = sitk.ReadImage('G:/.shortcut-targets-by-id/1L4UF0b3NvLVzjuyg84AcKuEyF-HRltzn/RM-RENAL_CUN&UPNA/DATA/Native/V09/mask_left/left.nii')
mask_right = sitk.ReadImage('G:/.shortcut-targets-by-id/1L4UF0b3NvLVzjuyg84AcKuEyF-HRltzn/RM-RENAL_CUN&UPNA/DATA/Native/V09/mask_right/right.nii')
ml = sitk.GetArrayFromImage(mask_left)
mr = sitk.GetArrayFromImage(mask_right)
# mask_arr = sitk.GetArrayFromImage(mask[:,:,0])
result = sitk.GetArrayFromImage(sitk.Image([96, 96], sitk.sitkUInt8))
result[ml > 0] = 1
result[mr > 0] = 1

unifiedmask = sitk.GetImageFromArray(result)
# unifiedmask.SetSpacing(mask[:,:,0].GetSpacing())
# unifiedmask.SetOrigin(mask[:,:,0].GetOrigin())
unifiedmask.SetSpacing(mask_left.GetSpacing())
unifiedmask.SetOrigin(mask_left.GetOrigin())

sitk.WriteImage(unifiedmask, 'Z:/pruebas/unified_V09.nii')