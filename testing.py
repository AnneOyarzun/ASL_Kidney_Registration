#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
import time
import csv
import pystrum.pynd.ndutils as nd
from data_generation import pca_generation

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  
torch.cuda.is_available()
torch.cuda.device_count()

def transform_image(img2transform, def_field, device, label = None): 
    '''
    Vamos a utilizar la función de STN de vxm.torch.layers. 
    Asegurarse de que sean Tensors. 
    '''
    # Convert img into Tensor--------------------------------
    img2trans_tensor = torch.from_numpy(img2transform).to(device).float().permute(0, 3, 1, 2)
    def_field = def_field[np.newaxis, ...] # add batch size
    def_field_tensor = torch.from_numpy(def_field).to(device).float()
    
    # Set transformer and apply def. field-----------------------------------------
    inshape = (96, 96)
    if label is None: 
        stn = vxm.torch.layers.SpatialTransformer(inshape)
        img_moved = stn(img2trans_tensor, def_field_tensor, interpolation_mode = 'nearest')
    else:
       stn = vxm.torch.layers.SpatialTransformer(inshape) 
       img_moved = stn(img2trans_tensor, def_field_tensor, interpolation_mode = 'nearest')
    
    img_moved_np = img_moved.detach().cpu().numpy().squeeze()

    return img_moved_np

def test_reg(input_moving, input_fixed, device, model, input_movingm=False, input_fixedm=False):
    input_moving = torch.from_numpy(input_moving).to(device).float().permute(0, 3, 1, 2)
    input_fixed = input_fixed.astype(np.float32)
    input_fixed = torch.from_numpy(input_fixed).to(device).float().permute(0, 3, 1, 2)
    
    if input_movingm:
        input_movingm = torch.from_numpy(input_movingm).to(device).float().permute(0, 3, 1, 2)
    if input_fixedm:
        input_fixedm = torch.from_numpy(input_fixedm).to(device).float().permute(0, 3, 1, 2)

    # Predict---------------------------------------------------
    moved, warp = model(input_moving, input_fixed, source_mask=False, target_mask=False, registration=True)

    # Save moved image------------------------------------------
    moved = moved.detach().cpu().numpy().squeeze()
    
    # Save warp-------------------------------------------------
    warp = warp.detach().cpu().numpy().squeeze()
    
    
    return moved, warp

def add_axis(arr): 
    arr = arr[np.newaxis, ...] # add batch size
    arr = arr[..., np.newaxis] # add feat 

    return arr

def create_new_serie(orig_serie, new_img, size): 
    '''
        1. borramos la primera imagen (ya está registrada, será la moved)
        2. Hacemos un append de la moved (al final, así tendremos siempre la siguiente imagen en img_series_ori[0])
    '''
    img_series = np.delete(orig_serie, [0], axis = 1)
    new_serie = np.append(img_series, np.reshape(new_img, (size, 1)), axis = 1)
    return new_serie

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    from voxelmorph
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def register(args, mode):
    # GPU SETTING -------------------------------------------
    if args['gpu'] and (args['gpu'] != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # LOAD MODEL --------------------------------------------
    if not os.path.exists(args['model']): 
        return
    else:
        model = vxm.networks.VxmDense.load(args['model'], device)
        model.to(device)
        model.eval()
    
    # Load data/cortex_masks --------------------------------------
    '''
        images → 32-bit float
        cortex_masks → 8-bit unsigned integer
    '''
    # if args['register_t1']:

    # else: 

    ori_images = sitk.ReadImage(args['data_path_pcasl']) 
    ori_images = sitk.Cast(ori_images, sitk.sitkFloat32) # in case of 16-bit-signed-integer
    ori_images = sitk.RescaleIntensity(ori_images, 0, 1)
    if args['labels_path']:
        ori_cortex_masks = sitk.ReadImage(args['labels_path'][0], sitk.sitkUInt8)
        ori_cortex_masks = sitk.GetArrayFromImage(ori_cortex_masks)
    
    ori_spacing = ori_images.GetSpacing()
    ori_origin = ori_images.GetOrigin()

    ori_images = sitk.GetArrayFromImage(ori_images)
    
    
    if args['labels_path']:
        ori_kidney_masks = sitk.ReadImage(args['labels_path'][1], sitk.sitkUInt8)
        ori_kidney_masks = sitk.GetArrayFromImage(ori_kidney_masks)
    
    # Training mode ---------------------------------------
    if mode is not 'full':
        if mode is 'control':
            images = ori_images[1:]
            images = images[::2]

            cortex_masks = ori_cortex_masks[1:]
            cortex_masks = cortex_masks[::2]

            if args['inference_kidney']: 
                kidney_masks = ori_kidney_masks[1:]
                kidney_masks = kidney_masks[::2]

        elif mode is 'label':
            images = ori_images[1:]
            images = images[1::2]

            cortex_masks = ori_cortex_masks[1:]
            cortex_masks = cortex_masks[1::2]

            if args['inference_kidney']: 
                kidney_masks = ori_kidney_masks[1:]
                kidney_masks = kidney_masks[1::2]
    else:
        if args['m0_reg'] == 1: # if M0 is to be exluded
            images = ori_images[1:]
            m0_img = ori_images[0,:,:]
            if args['labels_path']:
                cortex_masks = ori_cortex_masks[1:]
                kidney_masks = ori_kidney_masks[1:]
            
        else:
            images = ori_images
            if args['labels_path']:
                cortex_masks = ori_cortex_masks
                kidney_masks = ori_kidney_masks

    # Convert into array and transpose --------------------------------------------
    img_tras = np.transpose(images, (1,2,0)) 
    img_series_ori = np.reshape(img_tras, (img_tras.shape[0]*img_tras.shape[1], images.shape[0]))

    if args['labels_path']:
        cortex_masks[cortex_masks > 1] = 1 # cortex_masks are multilabel (1: right, 2: left). We are not distinguishing. 
        cortexmask_tras = np.transpose(cortex_masks, (1,2,0))
        cortexmask_series_ori = np.reshape(cortexmask_tras, (cortexmask_tras.shape[0]*cortexmask_tras.shape[1], cortex_masks.shape[0]))

        # kidneymasks_arr[kidneymasks_arr > 1] = 1 # cortex_masks are multilabel (1: right, 2: left). We are not distinguishing. 
        kidneymask_tras = np.transpose(kidney_masks, (1,2,0))
        kidneymask_series_ori = np.reshape(kidneymask_tras, (kidneymask_tras.shape[0]*kidneymask_tras.shape[1], kidney_masks.shape[0]))
        
    ##################################################################################
    ################################## REGISTRATION ##################################
    ##################################################################################

    # Timings ----------------------------------------
    elapsed_img = []
    elapsed_cortex = []
    elapsed_kidney = []
    neg_jacDet = []
    
    
    for imgs in range(0, images.shape[0]): 
        moving_img = images[imgs,:,:]
        moving_img = add_axis(moving_img)

        if args['labels_path']:
            moving_Cmask = cortex_masks[imgs,:,:]
            moving_Cmask = add_axis(moving_Cmask)

            moving_Kmask = kidney_masks[imgs,:,:]
            moving_Kmask = add_axis(moving_Kmask)

        if imgs == 0: # Serie de imgs es la original   
            # Templates ---------------------------------------------------
            fixed_img = pca_generation.template_pca(img_series_ori) 
            fixed_img = add_axis(fixed_img)
            # if args['labels_path']:
            #     fixed_mask = pca_generation.template_pca(cortexmask_series_ori, label=1)
            #     fixed_mask = add_axis(fixed_mask)

            # Inference and extract def. field (warp) --------------------------------------------------
            start_time = time.time()
            registered_img, warp = test_reg(moving_img, fixed_img, device, model, input_movingm=False, input_fixedm=False)
            elapsed_time = time.time() - start_time
            jacDet = jacobian_determinant(np.transpose(warp, (1, 2, 0)))

            num_negative_jacobians = np.sum(jacDet < 0)
            neg_jacDet.append(num_negative_jacobians)

            new_serie = create_new_serie(img_series_ori, registered_img, img_series_ori.shape[0])
            
            if args['save_warp']:
                if args['m0_reg'] == 0:
                    if not os.path.exists(args['result_path'] + 'with_M0/warps/'): 
                        os.makedirs(args['result_path'] + 'with_M0/warps/')
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'with_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'with_M0/warps/jacDet_' + str(imgs) + '.nii')
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'with_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'with_M0/warps/jacDet_' + str(imgs) + '.nii')
                
                if args['m0_reg'] == 1:
                    if not os.path.exists(args['result_path'] + 'without_M0/warps/'): 
                        os.makedirs(args['result_path'] + 'without_M0/warps/')
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'without_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'without_M0/warps/jacDet_' + str(imgs) + '.nii')
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'without_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'without_M0/warps/jacDet_' + str(imgs) + '.nii')
            
                
            # Inference on cortex_masks (cortex, kidney)
            if args['inference_cortex']: 
                t_C = time.time()
                registered_cortex = transform_image(moving_Cmask, warp, device, label=1)
                elapsed_cortex.append(time.time() - t_C)
                new_serie_cortex_mask = create_new_serie(cortexmask_series_ori, registered_cortex, img_series_ori.shape[0])
            
            if args['inference_kidney']:  
                t_K = time.time()
                registered_kidney = transform_image(moving_Kmask, warp, device, label=1)
                elapsed_kidney.append(time.time() - t_K)
                new_serie_kidney_mask = create_new_serie(kidneymask_series_ori, registered_kidney, img_series_ori.shape[0])            

        else:
            # Templates ---------------------------------------------------
            fixed_img = pca_generation.template_pca(new_serie) 
            fixed_img = add_axis(fixed_img)

            # if args['inference_cortex']:  
            #     fixed_mask = pca_generation.template_pca(new_serie_cortex_mask, label=1)
            #     fixed_mask = add_axis(fixed_mask)

            # Inference and extract def. field (warp) --------------------------------------------------
            t_img = time.time()
            registered_img, warp = test_reg(moving_img, fixed_img, device, model, input_movingm=False, input_fixedm=False)
            elapsed_img.append(time.time() - t_img)
            jacDet = jacobian_determinant(np.transpose(warp, (1, 2, 0)))
            num_negative_jacobians = np.sum(jacDet < 0)
            neg_jacDet.append(num_negative_jacobians)
            new_serie = create_new_serie(new_serie, registered_img, img_series_ori.shape[0])
            
            if args['save_warp']:
                if args['m0_reg'] == 0:
                    if not os.path.exists(args['result_path'] + 'with_M0/warps/'): 
                        os.makedirs(args['result_path'] + 'with_M0/warps/')
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'with_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'with_M0/warps/jacDet_' + str(imgs) + '.nii')
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'with_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'with_M0/warps/jacDet_' + str(imgs) + '.nii')
                    
                    # Save negative jacdet list
                    with open (args['result_path'] + 'with_M0/warps/neg_jacDet.csv', mode = 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow('negative jacobians')
                        for value in neg_jacDet:
                            writer.writerow([value])
                
                if args['m0_reg'] == 1:
                    if not os.path.exists(args['result_path'] + 'without_M0/warps/'): 
                        os.makedirs(args['result_path'] + 'without_M0/warps/')
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'without_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'without_M0/warps/jacDet_' + str(imgs) + '.nii')
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(warp), args['result_path'] + 'without_M0/warps/warp_' + str(imgs) + '.nii')
                        sitk.WriteImage(sitk.GetImageFromArray(jacDet), args['result_path'] + 'without_M0/warps/jacDet_' + str(imgs) + '.nii')
                    
                    # Save negative jacdet list
                    with open (args['result_path'] + 'without_M0/warps/neg_jacDet.csv', mode = 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow('negative jacobians')
                        for value in neg_jacDet:
                            writer.writerow([value])
            
            
            # Inference on cortex_masks (cortex, kidney) --------------------------
            if args['inference_cortex']: 
                t_C = time.time()
                registered_cortex = transform_image(moving_Cmask, warp, device)
                elapsed_cortex.append(time.time() - t_C)
                new_serie_cortex_mask = create_new_serie(new_serie_cortex_mask, registered_cortex, img_series_ori.shape[0]) 
            if args['inference_kidney']:
                t_K = time.time()
                registered_kidney = transform_image(moving_Kmask, warp, device)
                elapsed_kidney.append(time.time() - t_K)
                new_serie_kidney_mask = create_new_serie(new_serie_kidney_mask, registered_kidney, img_series_ori.shape[0])  

    
    # SAVE RESULTS (Registered imgs)
    new_shape = (images.shape[2], images.shape[1], images.shape[0])
    registered_serie = np.reshape(new_serie, new_shape)
    modifiedResult = sitk.GetImageFromArray(np.transpose(registered_serie, (2,0,1)))
    modifiedResult.SetSpacing(ori_spacing)
    modifiedResult.SetOrigin(ori_origin)
    res_result = sitk.RescaleIntensity(modifiedResult, 0, 255)
       
    if args['m0_reg'] == 0:
        if not os.path.exists(args['result_path'] + 'with_M0/'+ 'images/'): 
            os.makedirs(args['result_path'] + 'with_M0/'+ 'images/')
            sitk.WriteImage(modifiedResult, args['result_path'] + 'with_M0/' + 'images/' + args['result_filename']) 
            sitk.WriteImage(res_result, args['result_path'] + 'with_M0/' + 'images/result_res.nii') 

        else:
            sitk.WriteImage(modifiedResult, args['result_path'] + 'with_M0/' + 'images/' + args['result_filename']) 
            sitk.WriteImage(res_result, args['result_path'] + 'with_M0/' + 'images/result_res.nii')
        
    if args['m0_reg'] == 1:
        m0_img_array = add_axis(m0_img)
        c_l_serie = pca_generation.template_pca(new_serie)
        c_l_serie_array = add_axis(c_l_serie)
        registered_m0, _ = test_reg(m0_img_array, c_l_serie_array, device, model, input_movingm=False, input_fixedm=False)
        if not os.path.exists(args['result_path'] + 'without_M0/'+ 'images/'): 
            os.makedirs(args['result_path'] + 'without_M0/'+ 'images/')
            sitk.WriteImage(modifiedResult, args['result_path'] + 'without_M0/'+ 'images/' + args['result_filename'])
            sitk.WriteImage(modifiedResult, args['result_path'] + 'without_M0/'+ 'images/result_res.nii')
            sitk.WriteImage(sitk.GetImageFromArray(registered_m0), args['result_path'] + 'without_M0/' + 'images/' + 'reg_M0.nii')
        else:
            sitk.WriteImage(modifiedResult, args['result_path'] + 'without_M0/'+ 'images/' + args['result_filename'])
            sitk.WriteImage(modifiedResult, args['result_path'] + 'without_M0/'+ 'images/result_res.nii')
            sitk.WriteImage(sitk.GetImageFromArray(registered_m0), args['result_path'] + 'without_M0/' + 'images/' + 'reg_M0.nii')
    

    if args['inference_cortex']: 
        new_serie_cortex_mask[new_serie_cortex_mask < 1] = 0
        registered_serie_cortexmask = np.reshape(new_serie_cortex_mask, new_shape)
        modifiedResult_cortexmask = sitk.GetImageFromArray(np.transpose(registered_serie_cortexmask, (2,0,1)))
        modifiedResult_cortexmask.SetSpacing(ori_spacing)
        modifiedResult_cortexmask.SetOrigin(ori_origin)

        if args['m0_reg'] == 0:
            if not os.path.exists(args['result_path'] + 'with_M0/'+ 'masks/Cortex/'): 
                os.makedirs(args['result_path'] + 'with_M0/'+ 'masks/Cortex/')
            sitk.WriteImage(modifiedResult_cortexmask, args['result_path'] + 'with_M0/' + 'masks/Cortex/' + args['result_filename']) 
        
        if args['m0_reg'] == 1:
            if not os.path.exists(args['result_path'] + 'without_M0/'+ 'masks/Cortex/'): 
                os.makedirs(args['result_path'] + 'without_M0/'+ 'masks/Cortex/')
            sitk.WriteImage(modifiedResult_cortexmask, args['result_path'] + 'without_M0/' + 'masks/Cortex/' + args['result_filename']) 

    if args['inference_kidney']: 
        new_serie_kidney_mask[new_serie_kidney_mask < 1] = 0
        registered_serie_kidneymask = np.reshape(new_serie_kidney_mask, new_shape)
        modifiedResult_kidneymask = sitk.GetImageFromArray(np.transpose(registered_serie_kidneymask, (2,0,1)))
        modifiedResult_kidneymask.SetSpacing(ori_spacing)
        modifiedResult_kidneymask.SetOrigin(ori_origin)
        
        if args['m0_reg'] == 0:
            if not os.path.exists(args['result_path'] + 'with_M0/'+ 'masks/Kidney/'): 
                os.makedirs(args['result_path'] + 'with_M0/'+ 'masks/Kidney/')
            sitk.WriteImage(modifiedResult_kidneymask, args['result_path'] + 'with_M0/' + 'masks/Kidney/' + args['result_filename']) 
        
        if args['m0_reg'] == 1:
            if not os.path.exists(args['result_path'] + 'without_M0/'+ 'masks/Kidney/'): 
                os.makedirs(args['result_path'] + 'without_M0/'+ 'masks/Kidney/')
            sitk.WriteImage(modifiedResult_kidneymask, args['result_path'] + 'without_M0/' + 'masks/Kidney/' + args['result_filename']) 

    
if __name__ == '__main__': 
    
    # General parameters ------------------------------------------------------------------------------------
    main_path = 'Z:/RM_RENAL/Registration/'
    model_path = main_path + 'Voxelmorph/Native/Models/'
    result_path = main_path + 'Voxelmorph/Native/Results/'
    weigths = '1000.pt'

    # groups = ['NCC', 
    #           'NCC-DICE/w_0.9-0.1/', 
    #           'NCC-DICE/w_0.8-0.2/', 
    #           'NCC-TSNR/w_0.9-0.1/', 
    #           'NCC-TSNR/w_0.95-0.05/'
    #           ]
    
    # groups = [
    #           'NCC/'
    #         #   'NCC-DICE/w_0.9-0.1/', 
    #         #   'NCC-DICE/w_0.8-0.2/'
    #           ]

    groups = ['NCC-DICE/w_0.9-0.1/']

    #-------------------------------------------------------------------------------------------------
    
    # lambda_values = [0.01, 0.1, 0.5, 0.9, 1, 2]
    # lambda_values = [0.01]
    # lambda_values = [0.5, 0.9, 1, 2]
    lambda_values = [2]

    # subjects = ['Allograft', 'Native']
    subjects = ['Allograft']

    
    # for imgtype in subjects:
    #     for group in groups: 
    #         for lambda_value in lambda_values:
    #             main_model_path = model_path + group 
                
    #             model_names = [
    #                             'U-l' + str(lambda_value), 
    #                             'B-l' + str(lambda_value),
    #                         ]
                
    #             # Testing 
    #             with open('data_paths_' + imgtype + '.txt') as f:
    #                 studies = f.read().splitlines() 

    #             for nstudies in range(0, len(studies)):

    #                 for nmodel in model_names:
    #                     args = {            'imgType': imgtype, 
    #                                         'data_path_pcasl': main_path + 'Elastix/PCASL/' + imgtype + '/' + studies[nstudies] + 'MI/result.0.nii',
    #                                         # 'data_path_t1': main_path + 'Elastix/T1/' + imgtype + '/' + studies[nstudies] + 'MI/result.0.nii',
    #                                         'labels_path': [main_path + 'Elastix/PCASL/' + imgtype + '/' + studies[nstudies] + 'MI/masks/Cortex/result.nii', main_path + 'Elastix/PCASL/' + imgtype + '/' + studies[nstudies] + 'MI/masks/Kidney/result.nii'], 
    #                                         'model': main_model_path + nmodel + '/' + weigths,
    #                                         'result_path': result_path + group + '/' + nmodel + '/' + imgtype + '/' + studies[nstudies], 
    #                                         'result_filename': 'result.nii', 
    #                                         'm0_reg': 0, # 1: Exclude M0, 0: Include M0, 
    #                                         'register_t1': False,
    #                                         'gpu': torch.cuda.get_device_name(0), 
    #                                         'multichannel': False, 
    #                                         'resize': None, 
    #                                         'inference_cortex': True, 
    #                                         'inference_kidney': True, 
    #                                         'save_warp': True
    #                             }

    
    # args = {            
    #     'imgType':  'Native', 
    #                 'data_path_pcasl': 'Z:/RM_RENAL/Registration/Elastix/PCASL_T1/Allograft/TRACTRL02/01/S01/PCA2/S01/result.0.nii',
    #                 # 'data_path_t1': main_path + 'Elastix/T1/' + imgtype + '/' + studies[nstudies] + 'MI/result.0.nii',
    #                 'labels_path': None, 
    #                 'model': 'Z:/RM_RENAL/Registration/Voxelmorph/Native/Models/NCC-DICE/w_0.9-0.1/U-l1/1000.pt',
    #                 'result_path': 'Z:/pruebas/pcasl_t1/', 
    #                 'result_filename': 'result.nii', 
    #                 'm0_reg': 0, # 1: Exclude M0, 0: Include M0, 
    #                 'register_t1': False,
    #                 'gpu': torch.cuda.get_device_name(0), 
    #                 'multichannel': False, 
    #                 'resize': None, 
    #                 'inference_cortex': False, 
    #                 'inference_kidney': False, 
    #                 'save_warp': False
    #                             }
    # register(args, mode = 'full')

    subject = 'V12'
    result_path = 'Z:/RM_RENAL/Registration/Voxelmorph/PCASL_T1/Native/'
    

    args = {            
                    'imgType':  'Allograft', 
                    'data_path_pcasl': 'Z:/RM_RENAL/Registration/Unregistered_Data/PCASL_T1/Native/V12.nii',
                    # 'data_path_t1': main_path + 'Elastix/T1/' + imgtype + '/' + studies[nstudies] + 'MI/result.0.nii',
                    'labels_path': None, 
                    'model': 'Z:/RM_RENAL/Registration/Voxelmorph/Native/Models/NCC-DICE/w_0.9-0.1/U-l2/1000.pt',
                    'result_path': result_path, 
                    'result_filename': 'result.nii', 
                    'm0_reg': 0, # 1: Exclude M0, 0: Include M0, 
                    'register_t1': False,
                    'gpu': torch.cuda.get_device_name(0), 
                    'multichannel': False, 
                    'resize': None, 
                    'inference_cortex': False, 
                    'inference_kidney': False, 
                    'save_warp': False
                                }
    register(args, mode = 'full')










