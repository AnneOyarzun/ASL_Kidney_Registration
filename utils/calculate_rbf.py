import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import csv
# from plots import temporal
from utils import image_processing
# from plots import scale_bar

def compute_RBF(asl_serie, kidney_path, cortex_path, medulla_path, subject=None, calculate_median = False): 
    # Read images and masks
    asl_serie = sitk.GetArrayFromImage(sitk.ReadImage(asl_serie))
    kidney_masks = sitk.GetArrayFromImage(sitk.ReadImage(kidney_path))
    cortex_masks = sitk.GetArrayFromImage(sitk.ReadImage(cortex_path))
    if medulla_path:
        medulla_masks = sitk.GetArrayFromImage(sitk.ReadImage(medulla_path))


    if subject == 'Native':
        r_kidney_masks, l_kidney_masks = image_processing.label_right_left(kidney_masks)
        if calculate_median:
            kidneyMaskR = image_processing.calculate_median_img(r_kidney_masks[1:])
            kidneyMaskR[kidneyMaskR>=1] = 1
            kidneyMaskR[kidneyMaskR<1] = 0
            kidneyMaskL = image_processing.calculate_median_img(l_kidney_masks[1:])
            kidneyMaskL[kidneyMaskL>=1] = 1
            kidneyMaskL[kidneyMaskL<1] = 0
        else:
            kidneyMaskR = r_kidney_masks    
            kidneyMaskL = l_kidney_masks      
    
    if subject == 'Allograft':
        if calculate_median:
            kidneyMask = image_processing.calculate_median_img(kidney_masks[1:])
            kidneyMask[kidneyMask>=1] = 1
            kidneyMask[kidneyMask<1] = 0
        else:
            kidneyMask = kidney_masks

    # Cortex
    if subject == 'Native':
        r_cortex_masks, l_cortex_masks = image_processing.label_right_left(cortex_masks)
        if calculate_median: 
            cortexMaskR = image_processing.calculate_median_img(r_cortex_masks[1:])
            cortexMaskR[cortexMaskR>=1] = 1
            cortexMaskR[cortexMaskR<1] = 0
            cortexMaskL = image_processing.calculate_median_img(l_cortex_masks[1:])
            cortexMaskL[cortexMaskL>=1] = 1
            cortexMaskL[cortexMaskL<1] = 0
        else:
            cortexMaskR = r_cortex_masks
            cortexMaskL = l_cortex_masks
    
    if subject == 'Allograft':
        if calculate_median: 
            cortexMask = image_processing.calculate_median_img(cortex_masks[1:])
            cortexMask[cortexMask>=1] = 1
            cortexMask[cortexMask<1] = 0
        else:
            cortexMask = cortex_masks
        
    # # Medulla
    # if subject == 'Native':
    #     if calculate_median: 
    #         r_medulla_masks, l_medulla_masks = image_processing.label_right_left(medulla_masks)
    #         medullaMaskR = image_processing.calculate_median_img(r_medulla_masks[1:])
    #         medullaMaskR[medullaMaskR>=1] = 1
    #         medullaMaskR[medullaMaskR<1] = 0
    #         medullaMaskL = image_processing.calculate_median_img(l_medulla_masks[1:])
    #         medullaMaskL[medullaMaskL>=1] = 1
    #         medullaMaskL[medullaMaskL<1] = 0
    #     else:
    #         medullaMaskR = r_medulla_masks[1:]
    #         medullaMaskL = l_medulla_masks[1:]

    # if subject == 'Allograft':
    #     if calculate_median:
    #         medullaMask = image_processing.calculate_median_img(medulla_masks[1:])
    #     else:
    #         medullaMask = medulla_masks[1:]
    
    # Extract RBF values
    if subject == 'Native':
        (corticalAU_Right_PRE, corticalAU_Right_POST, 
        corticalAU_Left_PRE, corticalAU_Left_POST, 
        PWIs_PRE, PWIs_Right_POST, PWIs_Left_POST, 
        Averaged_PWIs_PRE, Averaged_PWIs_Right_POST, Averaged_PWIs_Left_POST,
        tsnr_right, tsnr_left) = image_processing.ASL_processing_native(asl_serie, cortexMaskR, cortexMaskL, filter=True, median_mask=calculate_median)
       
        # print('tsnr right', tsnr_right)
        # print('tsnr left', tsnr_left)
        M0 = asl_serie[0,:,:]
        rbf = rbf_computation(M0, Averaged_PWIs_Right_POST)
        if calculate_median:
            RightCortex_rbf = image_processing.compute_mean(rbf, cortexMaskR)
            LeftCortex_rbf = image_processing.compute_mean(rbf, cortexMaskL)
        # RightMedulla_rbf = image_processing.compute_mean(rbf, medullaMaskR)
        # LeftMedulla_rbf = image_processing.compute_mean(rbf, medullaMaskL)
        else:
            cortexMaskR = image_processing.calculate_median_img(r_cortex_masks[1:])
            cortexMaskR[cortexMaskR>=1] = 1
            cortexMaskR[cortexMaskR<1] = 0
            cortexMaskL = image_processing.calculate_median_img(l_cortex_masks[1:])
            cortexMaskL[cortexMaskL>=1] = 1
            cortexMaskL[cortexMaskL<1] = 0

            RightCortex_rbf = image_processing.compute_mean(rbf, cortexMaskR)
            LeftCortex_rbf = image_processing.compute_mean(rbf, cortexMaskL)

        return RightCortex_rbf, LeftCortex_rbf, tsnr_right, tsnr_left
    if subject == 'Allograft':
        (       corticalAU_PRE, corticalAU_POST, \
                PWIs_PRE, PWIs_POST, \
                Averaged_PWIs_PRE, Averaged_PWIs_POST, \
                tsnr
            ) = image_processing.ASL_processing_allograft(asl_serie, cortexMask, filter=True, median_mask=calculate_median)
       
        M0 = asl_serie[0,:,:]
        rbf = rbf_computation(M0, Averaged_PWIs_POST)

        if calculate_median:
            Cortex_rbf = image_processing.compute_mean(rbf, cortexMask)
        else:
            cortexMask = image_processing.calculate_median_img(cortex_masks[1:])
            cortexMask[cortexMask>=1] = 1
            cortexMask[cortexMask<1] = 0
            Cortex_rbf = image_processing.compute_mean(rbf, cortexMask)

        return Cortex_rbf, tsnr


def rbf_computation(M0, PWI): 
    '''
    PWIs_Right_POST lo hemos elegido como PWI (considerando que right y left es igual)
    '''
    lambda_val = 0.9
    delay = 0.058   
    pld = 1.200
    t1b = 1.650 
    alfa = 0.74 * 0.93 * 0.93 #%antes con 0.75
    tau = 1.600
   
    # RBF computation
    rbf = np.zeros((96, 96), dtype=np.float64)
    rbf= (6000 * lambda_val * PWI * np.exp(pld / t1b)) / (2 * alfa * t1b * M0 * (1 - np.exp(-tau / t1b)))
    rbf[rbf == 0] = np.nan

    # Delete very high values
    rbf_thres = rbf < 1000
    rbf_final = rbf * rbf_thres

    return rbf_final


