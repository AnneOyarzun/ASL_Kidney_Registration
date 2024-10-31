import SimpleITK as sitk
import numpy as np
import os
import cv2 as cv
from utils import image_processing

def calculate_pwi_serie(img_serie, mode = None): 
    # img_serie.astype(np.int16)
    if (img_serie.shape[0] % 2) == 0: # es par
        control_idx = range(1, img_serie.shape[0], 2)
        label_idx = range(0, img_serie.shape[0], 2)
    else:
        control_idx = range(2, img_serie.shape[0], 2)
        label_idx = range(1, img_serie.shape[0], 2)
    
    pwi_serie = sitk.Image([img_serie.shape[2], img_serie.shape[1], len(control_idx)], sitk.sitkFloat32)
    pwi_serie_arr = sitk.GetArrayFromImage(pwi_serie)
    for pairs in range(0, len(control_idx)):
        if mode == "inverted": 
            pwi_serie_arr[pairs,:,:] = img_serie[label_idx[pairs],:,:] - img_serie[control_idx[pairs],:,:]
        else:
            pwi_serie_arr[pairs,:,:] = img_serie[control_idx[pairs],:,:] - img_serie[label_idx[pairs],:,:]

    return pwi_serie_arr


def calculate_mean_img(images): 
    arr = np.array(np.mean(images, axis=(0)))
    return arr

def calculate_median_img(images): 
    arr = np.array(np.median(images, axis=(0)))
    return arr

def extract_avg_pwi(images, main_path, save=None): 
    images = images.astype(np.float32)
    pwi_serie = image_processing.calculate_pwi_serie(images)
    avg_pwi = image_processing.calculate_mean_img(pwi_serie)
    pwi_path = main_path
    avg_path = main_path + '/avg_Control_Label/'

    m0 = images[0,:,:]
    control_imgs = images[1:]
    control_imgs = control_imgs[1::2]
    label_imgs = images[1:]
    label_imgs = label_imgs[::2]

    avg_control = image_processing.calculate_mean_img(control_imgs)
    avg_label = image_processing.calculate_mean_img(label_imgs)

    substracted_avg = avg_control - avg_label

    avg_control_bym0 = cv.divide(avg_control, m0)
    avg_label_bym0 = cv.divide(avg_label, m0)
    substracted_avg_bym0 = cv.divide(substracted_avg, m0)
    
    if save:
        if not os.path.exists(pwi_path):
            os.makedirs(pwi_path)
        # if not os.path.exists(avg_path):
        #     os.makedirs(avg_path + 'controls/')
        #     os.makedirs(avg_path + 'labels/')
        #     os.makedirs(avg_path + 'substractions/')
        #     os.makedirs(avg_path + 'bym0/controls/')
        #     os.makedirs(avg_path + 'bym0/labels/')
        #     os.makedirs(avg_path + 'bym0/substractions/')

        sitk.WriteImage(sitk.GetImageFromArray(avg_pwi), pwi_path + 'pwi.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_control), pwi_path + 'controls.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_label), pwi_path + 'labels.nii')
        sitk.WriteImage(sitk.GetImageFromArray(substracted_avg), pwi_path + 'substractions.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_control_bym0), pwi_path + 'bym0_controls.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_label_bym0), pwi_path + 'bym0_labels.nii')
        sitk.WriteImage(sitk.GetImageFromArray(substracted_avg_bym0), pwi_path + 'bym0_substractions.nii')

        # # If testing one model per study..
        # sitk.WriteImage(sitk.GetImageFromArray(avg_pwi), pwi_path + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_control), avg_path + 'controls/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_label), avg_path + 'labels/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(substracted_avg), avg_path + 'substractions/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_control_bym0), avg_path + 'bym0/controls/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_label_bym0), avg_path + 'bym0/labels/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(substracted_avg_bym0), avg_path + 'bym0/substractions/' + str(id+1) + '.nii')

    return avg_pwi
    
    # elif image_group == 'Tested_Controls':
    #     avg_path = main_path + '/Voxelmorph/Native/Results/' + studies[nstudies] + main_modelname + '/' + loss_opt + '/' + experiment + '/' + image_group 
    #     avg_control = image_processing.calculate_mean_img(images)
    #     if not os.path.exists(avg_path + '/avg_controls/'):
    #         os.makedirs(avg_path + '/avg_controls/')
    #     sitk.WriteImage(sitk.GetImageFromArray(avg_control), avg_path + '/avg_controls/' + str(id+1) + '.nii')

    # elif image_group == 'Tested_Labels':
    #     avg_path = main_path + '/Voxelmorph/Native/Results/' + studies[nstudies] + main_modelname + '/' + loss_opt + '/' + experiment + '/' + image_group 
    #     avg_label = image_processing.calculate_mean_img(images)
    #     if not os.path.exists(avg_path + '/avg_labels/'):
    #         os.makedirs(avg_path + '/avg_labels')
    #     sitk.WriteImage(sitk.GetImageFromArray(avg_label), avg_path + '/avg_labels/' + str(id+1) + '.nii')
