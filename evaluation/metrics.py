import numpy as np
import cv2 as cv
import SimpleITK as sitk
import os
import csv
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from utils import computeBB
from utils import image_processing
from utils import detect_r_l
from evaluation import metrics
from utils import evaluation



def ncc(im1, im2):
    # Calculate the mean and standard deviation of each image
    mean1, std1 = cv.meanStdDev(im1)
    mean2, std2 = cv.meanStdDev(im2)

    # Calculate NCC
    ncc = np.sum((im1 - mean1) * (im2 - mean2)) / (im1.shape[0] * im1.shape[1] * std1 * std2)
    return ncc


def metric_MSSIM_NCC_allo(imgs, roi, win_size=7): 
    mean_ssim_full = []
    mean_ssim_masked = []
    mean_ncc_full = []
    mean_ncc_masked = []

    for i in range(0, imgs.shape[0]): # mismo size que reg
        ref_image = imgs[i,:,:]
        ref_roi = roi[i,:,:]
        filtered_positions = [index for index, _ in enumerate(imgs) if index != i]
        rest_images = imgs[filtered_positions]

        ssim_full = []
        ssim_masked = []
        ncc_full = []
        ncc_masked = []

        for excluded_imgs in range(0, len(rest_images)): 
            im1 = cv.normalize(ref_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            im1 = im1.astype(np.uint8)
            im2 = cv.normalize(rest_images[excluded_imgs,:,:], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            im2 = im2.astype(np.uint8)

            bb_coords = computeBB.extract_bboxes(ref_roi)

            # If the masks is small, reduce the win_size
            try:
                ssim_full.append(ssim(im1, im2, win_size=win_size)) # 7 o 9 ventana
                ncc_full.append(ncc(im1, im2))
            except ValueError: 
                pass
            try:
                ssim_masked.append(ssim(im1[bb_coords[1]: bb_coords[3], bb_coords[0]: bb_coords[2]], im2[bb_coords[1]: bb_coords[3], bb_coords[0]: bb_coords[2]], win_size=win_size))
                ncc_masked.append(ncc(im1[bb_coords[1]: bb_coords[3], bb_coords[0]: bb_coords[2]], im2[bb_coords[1]: bb_coords[3], bb_coords[0]: bb_coords[2]]))

            except ValueError:  
                pass
            
        # Patient-wise mean 
        mean_ssim_full.append(np.median(ssim_full))
        mean_ncc_full.append(np.median(ncc_full))
        mean_ssim_masked.append(np.median(ssim_masked))
        mean_ncc_masked.append(np.median(ncc_masked))

    total_ssim_full = np.median(mean_ssim_full)
    total_ssim_masked = np.median(mean_ssim_masked)
    total_ncc_full = np.median(mean_ncc_full)
    total_ncc_masked = np.median(mean_ncc_masked)

    return total_ssim_full, total_ssim_masked, total_ncc_full, total_ncc_masked

def metric_MSSIM_NCC_native(imgs, roi_R, roi_L, win_size=7): 
    mean_ssim_full = []
    mean_ssim_masked_R = []
    mean_ssim_masked_L = []
    mean_ncc_full = []
    mean_ncc_masked_R = []
    mean_ncc_masked_L = []


    for i in range(0, imgs.shape[0]): # mismo size que reg
        ref_image = imgs[i,:,:]
        ref_roi_R = roi_R[i,:,:]
        ref_roi_L = roi_L[i,:,:]
        filtered_positions = [index for index, _ in enumerate(imgs) if index != i]
        rest_images = imgs[filtered_positions]

        ssim_full = []
        ssim_masked_R = []
        ssim_masked_L = []
        ncc_full = []
        ncc_masked_R = []
        ncc_masked_L = []
        
        for excluded_imgs in range(0, len(rest_images)): 
            im1 = cv.normalize(ref_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            im1 = im1.astype(np.uint8)
            im2 = cv.normalize(rest_images[excluded_imgs,:,:], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            im2 = im2.astype(np.uint8)

            bb_coords_R = computeBB.extract_bboxes(ref_roi_R)
            bb_coords_L = computeBB.extract_bboxes(ref_roi_L)

            # If the masks is small, reduce the win_size
            try:
                ssim_full.append(ssim(im1, im2, win_size=win_size)) # 7 o 9 ventana
                ncc_full.append(ncc(im1, im2)) # 7 o 9 ventana

            except ValueError: 
                pass
            try:
                ssim_masked_R.append(ssim(im1[bb_coords_R[1]: bb_coords_R[3], bb_coords_R[0]: bb_coords_R[2]], im2[bb_coords_R[1]: bb_coords_R[3], bb_coords_R[0]: bb_coords_R[2]], win_size=win_size))
                ssim_masked_L.append(ssim(im1[bb_coords_L[1]: bb_coords_L[3], bb_coords_L[0]: bb_coords_L[2]], im2[bb_coords_L[1]: bb_coords_L[3], bb_coords_L[0]: bb_coords_L[2]], win_size=win_size))
                ncc_masked_R.append(ncc(im1[bb_coords_R[1]: bb_coords_R[3], bb_coords_R[0]: bb_coords_R[2]], im2[bb_coords_R[1]: bb_coords_R[3], bb_coords_R[0]: bb_coords_R[2]]))
                ncc_masked_L.append(ncc(im1[bb_coords_L[1]: bb_coords_L[3], bb_coords_L[0]: bb_coords_L[2]], im2[bb_coords_L[1]: bb_coords_L[3], bb_coords_L[0]: bb_coords_L[2]]))
            except ValueError:  
                pass
            
        # Patient-wise mean 
        mean_ssim_full.append(np.median(ssim_full))
        mean_ssim_masked_R.append(np.median(ssim_masked_R))
        mean_ssim_masked_L.append(np.median(ssim_masked_L))
        mean_ncc_full.append(np.median(ncc_full))
        mean_ncc_masked_R.append(np.median(ncc_masked_R))
        mean_ncc_masked_L.append(np.median(ncc_masked_L))

    total_ssim_full = np.median(mean_ssim_full)
    total_ssim_masked_R = np.median(mean_ssim_masked_R)
    total_ssim_masked_L = np.median(mean_ssim_masked_L)
    total_ncc_full = np.median(mean_ncc_full)
    total_ncc_masked_R = np.median(mean_ncc_masked_R)
    total_ncc_masked_L = np.median(mean_ncc_masked_L)

    return total_ssim_full, total_ssim_masked_R, total_ssim_masked_L, total_ncc_full, total_ncc_masked_R, total_ncc_masked_L


def dice_score(image1, image2):
    intersection = np.logical_and(image1, image2).sum()
    union = image1.sum() + image2.sum()
    dice = (2.0 * intersection) / union if union > 0 else 1.0
    return dice

def groupwise_dice(images):
    num_images = len(images)
    dice_scores = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i+1, num_images):
            dice = dice_score(images[i], images[j])
            dice_scores[i, j] = dice_scores[j, i] = dice

    return dice_scores

def groupwise_dice_refImage(images, ref):
    num_images = len(images)
    dice_scores = np.zeros(num_images)

    for i in range(num_images):
            dice = dice_score(ref, images[i])
            dice_scores[i] = dice

    return dice_scores


def save_positions_to_csv(positions, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Slice", "Position"])
        for position in positions:
            writer.writerow(position)

def erode_mask(mask_to_erode, kernel_size=2):
    # Define the structuring element (kernel) for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform erosion
    eroded_mask = cv.erode(mask_to_erode, kernel, iterations=1)
    return eroded_mask

def calculate_tsnr_allo(images, cortex_masks, cortexMaskAvg, pwi_path, image_group=None, filter=False, save_filtered_pos=False, filtered_pos=False): 
    if image_group == 'Tested_Full' or image_group == 'Tested_FullnoM0' or image_group == None:
        pwi_serie = image_processing.calculate_pwi_serie(images)
    elif image_group == 'Tested_Controls' or image_group == 'Tested_Labels': 
        pwi_serie = images

    control_idx = range(2, images.shape[0], 2)
    label_idx = range(1, images.shape[0], 2)

    # erosion
    eroded_mask = erode_mask(cortexMaskAvg, 2)
    
    tsnr_cortex_pre = []
    
    for pwi_imgs in range(0, pwi_serie.shape[0]): 
        # tsnr_cortex_pre.append(evaluation.compute_mean(pwi_serie[pwi_imgs, :, :], eroded_mask))
        eroded_cortexControl = erode_mask(cortex_masks[control_idx[pwi_imgs], :, :])
        eroded_cortexLabel = erode_mask(cortex_masks[label_idx[pwi_imgs], :, :])
        tsnr_cortex_pre.append(evaluation.compute_mean(images[control_idx[pwi_imgs], :, :], eroded_cortexControl) - evaluation.compute_mean(images[label_idx[pwi_imgs], :, :], eroded_cortexLabel))
        # tsnr_cortex_pre.append(evaluation.compute_mean(images[control_idx[pwi_imgs], :, :], cortex_masks[control_idx[pwi_imgs], :, :]) - evaluation.compute_mean(images[label_idx[pwi_imgs], :, :], cortex_masks[label_idx[pwi_imgs], :, :]))

    if filter:
        if not filtered_pos:
            # Post-process
            pos_threshold = np.nanmean(tsnr_cortex_pre) + (2 * np.nanstd(tsnr_cortex_pre))
            neg_threshold = np.nanmean(tsnr_cortex_pre) - (2 * np.nanstd(tsnr_cortex_pre))

            mean_filtered = tsnr_cortex_pre.copy()

            filtered_image = pwi_serie.copy()
            
            # List to store positions that meet the condition
            positions_to_save = []
            for i in range(0, pwi_serie.shape[0]): 
                if not neg_threshold < tsnr_cortex_pre[i] < pos_threshold:
                    mean_filtered[i] = np.nan
                    filtered_image[i,:,:] = np.nan
                    # Append the position to the list
                    # positions_to_save.append([i, np.nan])
            
            mean_slice_filtered = evaluation.compute_tsnr(mean_filtered)
            filtered_pwi = np.nanmean(filtered_image, axis=0)  # Compute mean along the first dimension (time)
            filtered_pwi_sd = np.nanstd(filtered_image, axis=0)
            sitk.WriteImage(sitk.GetImageFromArray(filtered_pwi), pwi_path + 'filtered_pwi.nii')
            sitk.WriteImage(sitk.GetImageFromArray(filtered_pwi_sd), pwi_path + 'filtered_pwiSD.nii')
            if save_filtered_pos:
                save_positions_to_csv(positions_to_save, pwi_path + 'pos_filtered.csv')

            return abs(np.asarray(mean_slice_filtered)), filtered_pwi
        else:
            try:
                pos2delete = pd.read_csv(filtered_pos, delimiter=',')
                pos2delete = pos2delete.dropna(axis=1)
                mean_filtered = tsnr_cortex_pre.copy()
                filtered_image = pwi_serie.copy()
                for pos in range(0, len(pos2delete)):
                    mean_filtered[pos2delete.at[pos, 'Slice']] = np.nan
                    filtered_image[pos2delete.at[pos, 'Slice'],:,:] = np.nan 
                mean_slice_filtered = evaluation.compute_tsnr(mean_filtered)
                filtered_pwi = np.nanmean(filtered_image, axis=0)
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwi), pwi_path + 'filtered_pwi.nii')

            except pd.errors.EmptyDataError:
                mean_slice_filtered = evaluation.compute_tsnr(mean_filtered)
                filtered_pwi = np.nanmean(pwi_serie, axis=0)  # Compute mean along the first dimension (time)
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwi), pwi_path + 'filtered_pwi.nii')

            return abs(np.asarray(mean_slice_filtered)), filtered_pwi
    else:
        mean_slice_filtered = evaluation.compute_tsnr(tsnr_cortex_pre)
        nonfiltered_pwi = np.nanmean(pwi_serie, axis=0)  # Compute mean along the first dimension (time)
        sitk.WriteImage(sitk.GetImageFromArray(nonfiltered_pwi), pwi_path + 'nonfiltered_pwi.nii')
        return abs(np.asarray(mean_slice_filtered)), nonfiltered_pwi

def calculate_tsnr_native(images, cortexR, cortexL, cortexMaskMedian_R, cortexMaskMedian_L, pwi_path, image_group=None, filter=filter, save_filtered_pos=False,filtered_pos=False):
   # Pre-filtered
    if image_group == 'Tested_Full' or image_group == 'Tested_FullnoM0' or image_group == None:
        pwi_serie = image_processing.calculate_pwi_serie(images)
    elif image_group == 'Tested_Controls' or image_group == 'Tested_Labels': 
        pwi_serie = images

    control_idx = range(2, images.shape[0], 2)
    label_idx = range(1, images.shape[0], 2)

    # erosion
    eroded_mask_L = erode_mask(cortexMaskMedian_L, 2)
    eroded_mask_R = erode_mask(cortexMaskMedian_R, 2)

    mean_kidney_right = []
    mean_kidney_left = []


    for pwi_imgs in range(0, pwi_serie.shape[0]): 
        # mean_kidney_left.append(evaluation.compute_mean(pwi_serie[pwi_imgs, :, :], eroded_mask_L))
        # mean_kidney_right.append(evaluation.compute_mean(pwi_serie[pwi_imgs, :, :], eroded_mask_R))
        eroded_cortexControlR = erode_mask(cortexR[control_idx[pwi_imgs], :, :])
        eroded_cortexLabelR = erode_mask(cortexR[label_idx[pwi_imgs], :, :])
        eroded_cortexControlL = erode_mask(cortexL[control_idx[pwi_imgs], :, :])
        eroded_cortexLabelL = erode_mask(cortexL[label_idx[pwi_imgs], :, :])

        # mean_kidney_right.append(evaluation.compute_mean(images[control_idx[pwi_imgs], :, :], cortexR[control_idx[pwi_imgs], :, :]) - evaluation.compute_mean(images[label_idx[pwi_imgs], :, :], cortexR[label_idx[pwi_imgs], :, :]))
        # mean_kidney_left.append(evaluation.compute_mean(images[control_idx[pwi_imgs], :, :], cortexL[control_idx[pwi_imgs], :, :]) - evaluation.compute_mean(images[label_idx[pwi_imgs], :, :], cortexL[label_idx[pwi_imgs], :, :]))

        mean_kidney_right.append(evaluation.compute_mean(images[control_idx[pwi_imgs], :, :], eroded_cortexControlR) - evaluation.compute_mean(images[label_idx[pwi_imgs], :, :], eroded_cortexLabelR))
        mean_kidney_left.append(evaluation.compute_mean(images[control_idx[pwi_imgs], :, :], eroded_cortexControlL) - evaluation.compute_mean(images[label_idx[pwi_imgs], :, :], eroded_cortexLabelL))

    if filter:
        if not filtered_pos: # Unregistered serie
            # Post-processing of tsnr ------------------------------------------------------------
            pos_Left_threshold = np.nanmean(mean_kidney_left) + (2 * np.nanstd(mean_kidney_left))
            neg_Left_threshold = np.nanmean(mean_kidney_left) - (2 * np.nanstd(mean_kidney_left))
            pos_Right_threshold = np.nanmean(mean_kidney_right) + (2 * np.nanstd(mean_kidney_right))
            neg_Right_threshold = np.nanmean(mean_kidney_right) - (2 * np.nanstd(mean_kidney_right))

            mean_left_filtered = mean_kidney_left.copy()
            mean_right_filtered = mean_kidney_right.copy()

            filtered_imageR = pwi_serie.copy()
            filtered_imageL = pwi_serie.copy()
            positions_to_save_R = []
            positions_to_save_L = []
            for i in range(0, pwi_serie.shape[0]): 
                if not neg_Left_threshold < mean_kidney_left[i] < pos_Left_threshold:
                    mean_left_filtered[i] = np.nan
                    filtered_imageL[i,:,:] = np.nan
                    positions_to_save_R.append([i, np.nan])
                if not neg_Right_threshold < mean_kidney_right[i] < pos_Right_threshold:
                    mean_right_filtered[i] = np.nan
                    filtered_imageR[i,:,:] = np.nan
                    positions_to_save_L.append([i, np.nan])
            
            mean_left_slice_filtered = evaluation.compute_tsnr(mean_left_filtered)
            mean_right_slice_filtered = evaluation.compute_tsnr(mean_right_filtered)

            filtered_pwiR = np.nanmean(filtered_imageR, axis=0)
            filtered_pwiL = np.nanmean(filtered_imageL, axis=0)
            filtered_pwiRsd = np.nanstd(filtered_imageR, axis=0)
            filtered_pwiLsd = np.nanstd(filtered_imageL, axis=0)
            sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiR), pwi_path + 'filtered_pwiR.nii')
            sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiL), pwi_path + 'filtered_pwiL.nii')
            sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiRsd), pwi_path + 'filtered_pwiRsd.nii')
            sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiLsd), pwi_path + 'filtered_pwiLsd.nii')
            
            if save_filtered_pos:
                save_positions_to_csv(positions_to_save_R, pwi_path + 'pos_filtered_R.csv')
                save_positions_to_csv(positions_to_save_L, pwi_path + 'pos_filtered_L.csv')

            return (
                abs(np.asarray(mean_right_slice_filtered)), 
                abs(np.asarray(mean_left_slice_filtered)), 
                filtered_pwiR, filtered_pwiL
            )
    
        else:
            try:
                pos2deleteR = pd.read_csv(filtered_pos[0], delimiter=',')
                pos2deleteR = pos2deleteR.dropna(axis=1)
                pos2deleteL = pd.read_csv(filtered_pos[1], delimiter=',')
                pos2deleteL = pos2deleteL.dropna(axis=1)

                mean_filteredR = mean_kidney_right.copy()
                mean_filteredL = mean_kidney_left.copy()
                filtered_imageR = pwi_serie.copy()
                filtered_imageL = pwi_serie.copy()

                for posR in range(0, len(pos2deleteR)):
                    mean_filteredR[pos2deleteR.at[posR, 'Slice']] = np.nan
                    filtered_imageR[pos2deleteR.at[posR, 'Slice'],:,:] = np.nan 
                mean_slice_filteredR = evaluation.compute_tsnr(mean_filteredR)
                filtered_pwiR = np.nanmean(filtered_imageR, axis=0)
                filtered_pwiR_sd = np.nanstd(filtered_imageR, axis=0)
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiR), pwi_path + 'filtered_pwiR.nii')
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiR_sd), pwi_path + 'filtered_pwiR_sd.nii')

                for posL in range(0, len(pos2deleteL)):
                    mean_filteredL[pos2deleteL.at[posL, 'Slice']] = np.nan
                    filtered_imageL[pos2deleteL.at[posL, 'Slice'],:,:] = np.nan 
                mean_slice_filteredL = evaluation.compute_tsnr(mean_filteredL)
                filtered_pwiL = np.nanmean(filtered_imageL, axis=0)
                filtered_pwiL_sd = np.nanstd(filtered_imageL, axis=0)
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiL), pwi_path + 'filtered_pwiL.nii')
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiL_sd), pwi_path + 'filtered_pwiL_sd.nii')

            except pd.errors.EmptyDataError:
                mean_slice_filteredR = evaluation.compute_tsnr(mean_filteredR)
                mean_slice_filteredL = mean_slice_filteredR
                filtered_pwiR = np.nanmean(pwi_serie, axis=0) 
                filtered_pwiL = filtered_pwiR
                sitk.WriteImage(sitk.GetImageFromArray(filtered_pwiR), pwi_path + 'filtered_pwi_both.nii')
                
            return abs(np.asarray(mean_slice_filteredR)), abs(np.asarray(mean_slice_filteredL)), filtered_pwiR, filtered_pwiL


    else:
        tsnr_mean_kidney_left = evaluation.compute_tsnr(mean_kidney_left)
        tsnr_mean_kidney_right = evaluation.compute_tsnr(mean_kidney_right)

        nonfiltered_pwi = np.nanmean(pwi_serie, axis=0) 
        sitk.WriteImage(sitk.GetImageFromArray(nonfiltered_pwi), pwi_path + 'nonfiltered_pwi.nii')
        
        return (
            abs(np.asarray(tsnr_mean_kidney_right)), 
            abs(np.asarray(tsnr_mean_kidney_left)), 
            nonfiltered_pwi,
        )


def compute_metrics_native(img_path, cortex_path, kidney_path, pwi_path, image_group = None, filter=False, save_filtered_pos=False, filtered_pos=False):
    # Data loading
    images = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    images = cv.normalize(images, None, 0, 255, cv.NORM_MINMAX) 

    cortex_masks = sitk.GetArrayFromImage(sitk.ReadImage(cortex_path))
    kidney_masks = sitk.GetArrayFromImage(sitk.ReadImage(kidney_path))
    r_cortex_masks, l_cortex_masks = detect_r_l.label_right_left(cortex_masks)
    r_kidney_masks, l_kidney_masks = detect_r_l.label_right_left(kidney_masks)

    cortexMaskMedian = image_processing.calculate_median_img(cortex_masks[1:])
    cortexMaskMedian[cortexMaskMedian>=1] = 1
    cortexMaskMedian[cortexMaskMedian<1] = 0

    cortexMaskMedianR = image_processing.calculate_median_img(r_cortex_masks[1:])
    cortexMaskMedianR[cortexMaskMedianR>=1] = 1
    cortexMaskMedianR[cortexMaskMedianR<1] = 0

    cortexMaskMedianL = image_processing.calculate_median_img(l_cortex_masks[1:])
    cortexMaskMedianL[cortexMaskMedianL>=1] = 1
    cortexMaskMedianL[cortexMaskMedianL<1] = 0

    kidneyMaskMedian = image_processing.calculate_median_img(kidney_masks[1:])
    kidneyMaskMedian[kidneyMaskMedian>=1] = 1
    kidneyMaskMedian[kidneyMaskMedian<1] = 0

    kidneyMaskMedianR = image_processing.calculate_median_img(r_kidney_masks[1:])
    kidneyMaskMedianR[kidneyMaskMedianR>=1] = 1
    kidneyMaskMedianR[kidneyMaskMedianR<1] = 0

    kidneyMaskMedianL = image_processing.calculate_median_img(l_kidney_masks[1:])
    kidneyMaskMedianL[kidneyMaskMedianL>=1] = 1
    kidneyMaskMedianL[kidneyMaskMedianL<1] = 0

    # Metric 1: MSSIM -------------------------------------------- 
    MSSIM_Full, MSSIM_Right, MSSIM_Left, NCC_Full, NCC_Right, NCC_Left = metrics.metric_MSSIM_NCC_native(images, r_kidney_masks, l_kidney_masks, win_size=9)

    # Metric 2: TSNR
    TSNR_Right, TSNR_Left, pwi_imgR, pwi_imgL = metrics.calculate_tsnr_native(images, r_cortex_masks, l_cortex_masks, cortexMaskMedianR, cortexMaskMedianL, pwi_path, image_group, filter=filter, save_filtered_pos=save_filtered_pos,filtered_pos=filtered_pos)

    # Metric 3: DICE
        # 1) Cortex
    dices_cortex = metrics.groupwise_dice_refImage(cortex_masks, cortexMaskMedian)
    DICE_Cortex_BothRL = np.median(dices_cortex)
    r_dices_cortex = metrics.groupwise_dice_refImage(r_cortex_masks, cortexMaskMedianR)
    DICE_Cortex_Right = np.median(r_dices_cortex)
    l_dices_cortex = metrics.groupwise_dice_refImage(l_cortex_masks, cortexMaskMedianL)
    DICE_Cortex_Left = np.median(l_dices_cortex)

        # 2) Kidney
    dices_kidney = metrics.groupwise_dice_refImage(kidney_masks,kidneyMaskMedian)
    DICE_Kidney_BothRL = np.median(dices_kidney)
    r_dices_kidney = metrics.groupwise_dice_refImage(r_kidney_masks, kidneyMaskMedianR)
    DICE_Kidney_Right = np.median(r_dices_kidney)
    l_dices_kidney = metrics.groupwise_dice_refImage(l_kidney_masks, kidneyMaskMedianL)
    DICE_Kidney_Left = np.median(l_dices_kidney)

    # Metric 4: PWI signal on cortex --------------------------------------------------------------------
    PWISIG_Cortex_Right = evaluation.compute_mean(pwi_imgR, cortexMaskMedianR)
    PWISIG_Cortex_Left = evaluation.compute_mean(pwi_imgL, cortexMaskMedianL)

    return MSSIM_Full, MSSIM_Right, MSSIM_Left, NCC_Full, NCC_Right, NCC_Left, TSNR_Right, TSNR_Left, DICE_Cortex_BothRL, DICE_Cortex_Right, DICE_Cortex_Left, DICE_Kidney_BothRL, DICE_Kidney_Right, DICE_Kidney_Left, PWISIG_Cortex_Right, PWISIG_Cortex_Left

def compute_metrics_allograft(img_path, cortex_path, kidney_path, pwi_path, image_group=None, filter=None, save_filtered_pos=False, filtered_pos=None): 
    # Data loading
    images = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    images = images.astype(np.float32)
    images = cv.normalize(images, None, 0, 255, cv.NORM_MINMAX) 
    cortex_masks = sitk.GetArrayFromImage(sitk.ReadImage(cortex_path))
    kidney_masks = sitk.GetArrayFromImage(sitk.ReadImage(kidney_path))

    cortexMaskMedian = image_processing.calculate_median_img(cortex_masks[1:])
    cortexMaskMedian[cortexMaskMedian>=1] = 1
    cortexMaskMedian[cortexMaskMedian<1] = 0

    kidneyMaskMedian = image_processing.calculate_median_img(kidney_masks[1:])
    kidneyMaskMedian[kidneyMaskMedian>=1] = 1
    kidneyMaskMedian[kidneyMaskMedian<1] = 0

    # Metric 1: MSSIM & NCC -------------------------------------------- 
    MSSIM_Full, MSSIM_Kidney, NCC_Full, NCC_Kidney = metrics.metric_MSSIM_NCC_allo(images, kidney_masks, win_size=9)

    # Metric 2: TSNR ----------------------------------------------
    TSNR_Cortex, pwi_img = metrics.calculate_tsnr_allo(images, cortex_masks, cortexMaskMedian, pwi_path, image_group=image_group,filter=filter, save_filtered_pos=save_filtered_pos, filtered_pos=filtered_pos)

    # Metric 3: DICE -------------------------------------------------------------------------------------
        # 1) Cortex
    dices_cortex = metrics.groupwise_dice_refImage(cortex_masks, cortexMaskMedian)
    DICE_Cortex = np.median(dices_cortex)

        # 2) Kidney
    dices_kidney = metrics.groupwise_dice_refImage(kidney_masks, kidneyMaskMedian)
    DICE_Kidney = np.median(dices_kidney)
    
    # Metric 4: PWI signal on cortex --------------------------------------------------------------------
    PWISIG_Cortex = evaluation.compute_mean(pwi_img, cortexMaskMedian)

    return MSSIM_Full, MSSIM_Kidney, NCC_Full, NCC_Kidney, TSNR_Cortex, DICE_Cortex, DICE_Kidney, PWISIG_Cortex