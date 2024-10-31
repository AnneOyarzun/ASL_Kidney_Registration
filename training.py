import os
from os import listdir
from os.path import isfile, join
from os.path import isfile, join
import matplotlib.pyplot as plt
import time
import math
import cv2 as cv
import numpy as np
import SimpleITK as sitk
import random
from tkinter import N
import numpy as np
import torch
import random
import torch.optim as optim
from torch.optim import lr_scheduler
import SimpleITK as sitk
import shutil
from natsort import natsorted
from tkinter.filedialog import askopenfilename, askdirectory
import cv2 as cv
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import neptune.new as neptune
run = neptune.init_run(project='aod7/VoxelMorph-tsnr', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2Q3YjZlZi03NjUzLTRlMWUtYmQ5Mi1kYzA2NDMzZjFhOGQifQ==')


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import natsort
from utils import tictoc
print(torch.version.cuda)
from utils import preprocessing
from utils import data_augmentation_tools

# -------------------------------------------------------------





def custom_augmentation(image, label, num_augmentations_per_type=5):
    augmented_images = []
    augmented_labels = []

    for _ in range(num_augmentations_per_type):
        # Randomly rotate the image within the range of -15 to +15 degrees
        angle = np.random.uniform(-10, 10)
        image_rotated = data_augmentation_tools.rotate_image_opencv(image, angle)
        label_rotated = data_augmentation_tools.rotate_image_opencv(label, angle)

        # plt.imshow(image_rotated, cmap='gray')
        # plt.imshow(label_rotated, cmap='jet', alpha=0.5)

        augmented_images.append(image_rotated)
        augmented_labels.append(label_rotated)

        # Apply random translation transformations
        translation_x = np.random.uniform(-5, 5)
        translation_y = np.random.uniform(-5, 5)
        image_translated = data_augmentation_tools.translate_image_opencv(image, translation_x, translation_y)
        label_translated = data_augmentation_tools.translate_image_opencv(label, translation_x, translation_y)

        augmented_images.append(image_translated)
        augmented_labels.append(label_translated)


    return augmented_images, augmented_labels


def train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full'): 
    
    # TRAIN DATA FOLDER
    if os.path.exists(trainData_path):
        shutil.rmtree(trainData_path)
    os.makedirs(trainData_path)
    if os.path.exists(trainLabel_path):
        shutil.rmtree(trainLabel_path)
    os.makedirs(trainLabel_path)


    # PRE-PREPROCESSING REGISTRATION ------------------------------------
    images = sitk.ReadImage(args['data_path'])
    if args['m0_reg'] == 1:
        images = images[:,:,1:images.GetSize()[2]]
    else:
        images = images

    if net_hyperparameters['image_loss'] == 'dual_ncc_tsnr': 
        label_path = args['labels_path_cortex']

    elif net_hyperparameters['image_loss'] == 'dual_ncc_dice':
        label_path = args['labels_path_kidney']
    else:
        label_path = args['labels_path_cortex'] # vale cualquiera

    masks = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    if args['imgType'] == "Native": 
        labels = np.zeros(masks.shape, dtype=np.uint8)
        labels[masks > 0] = 1 # unificamos r/l si es native

    else:
        labels = masks

    if mode is not 'full':
        if mode is 'control':
            ids = range(2, images.GetSize()[2], 2)
    
        elif mode is 'label':
            ids = range(1, images.GetSize()[2], 2)
    else:
        ids = range(0, images.GetSize()[2])


    for imgs in range(0, len(ids)):
        img = images[:,:, ids[imgs]]
        
        if net_hyperparameters['aug']:
            res_img = sitk.RescaleIntensity(img, 0, 1)
            augmented_imgs, augmented_labels = custom_augmentation(sitk.GetArrayFromImage(res_img), labels[imgs,:,:])

            for i, augmented_img in enumerate(augmented_imgs):
                # Convert augmented_img back to SimpleITK format
                augmented_img_sitk = sitk.GetImageFromArray(augmented_img)
                
            # Save augmented image and label with a unique index
            sitk.WriteImage(augmented_img_sitk, trainData_path + str(imgs + 1) + f'_aug_{i}.nii')
            sitk.WriteImage(sitk.GetImageFromArray(augmented_labels[i]), trainLabel_path + str(imgs + 1) + f'_aug_{i}.nii')
        else: 
            img = sitk.RescaleIntensity(img, 0, 1)
            sitk.WriteImage(img, trainData_path + str(imgs + 1) + '.nii')        
            sitk.WriteImage(sitk.GetImageFromArray(labels[imgs,:,:]), trainLabel_path + str(imgs + 1) + '.nii')


    f = open("img_list_groupwise.txt", "w+")
    f.truncate(0)
    m = open("mask_list_groupwise.txt", "w+")
    m.truncate(0)
    for i in range(0, len(natsorted(os.listdir(trainData_path)))): 
         f.write(trainData_path + natsorted(os.listdir(trainData_path))[i] + '\n')
         m.write(trainLabel_path + natsorted(os.listdir(trainLabel_path))[i] + '\n')
    f.close()
    m.close()
    img_list = 'img_list_groupwise.txt'
    mask_list = 'mask_list_groupwise.txt'

    # READ FILE LIST ---------------------------------------------------------------
    train_files = vxm.py.utils.read_file_list(img_list, prefix = net_hyperparameters['img_prefix'],
                                            suffix = net_hyperparameters['img_suffix'])
    train_masks = vxm.py.utils.read_file_list(mask_list, prefix = net_hyperparameters['img_prefix'],
                                        suffix = net_hyperparameters['img_suffix'])

    assert len(train_files) > 0, 'Could not find any training data.'
    
    # add_feat_axis = not net_hyperparameters['multichannel']
    
    # GPU SETTING-------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2809)
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
        ngpus = torch.cuda.device_count()
        print("Using {} GPU(s)...".format(ngpus))

    # # Early stopping
    # early_stopping_patience = 50  # Number of epochs without improvement to wait before stopping
    # best_training_loss = float('inf')
    # epochs_without_improvement = 0
    
    # GENERATOR -----------------------------------------------------------------------
    generator = vxm.generators.scan_to_scan(
            train_files, bidir=net_hyperparameters['bidir'], batch_size=net_hyperparameters['batch_size'], seg_names=train_masks)

    # Extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1] # shape of the imgs
    os.makedirs(model_dir, exist_ok=True)

    # LOAD/SET MODEL--------------------------------------------------------------------
    # Network layers (default)
    enc_nf = [16, 32, 32, 32] 
    dec_nf = [32, 32, 32, 32, 32, 16, 16] 

    # enc_nf = [16, 16, 32, 32] 
    # dec_nf = [32, 32, 16, 16] 

    if net_hyperparameters['load_model']:
        # Load initial model (if specified)
        model = vxm.networks.VxmDense.load(net_hyperparameters['load_model'], device)
    else:
        # Otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir= net_hyperparameters['bidir'],
            int_steps= net_hyperparameters['int_steps'],
            int_downsize= net_hyperparameters['int_downsize']
            )
        
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        model = model.to(device)
    model.train()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = net_hyperparameters['lr'])

    # Loss functions
    if net_hyperparameters['image_loss'] == 'dual_ncc_tsnr': 
        ncc_weight = net_hyperparameters['ncc_weight']
        tsnr_weight = net_hyperparameters['tsnr_weight']
        image_loss_func = vxm.losses.Duo_NCC_TSNR(net_hyperparameters['windowSize'], ncc_weight, tsnr_weight).loss
    
    elif net_hyperparameters['image_loss'] == 'dual_ncc_dice': 
        ncc_weight = net_hyperparameters['ncc_weight']
        dice_weight = net_hyperparameters['dice_weight']
        image_loss_func = vxm.losses.Duo_NCC_DICE(net_hyperparameters['windowSize'], ncc_weight, dice_weight).loss
    
    else:
        image_loss_func = vxm.losses.NCC(net_hyperparameters['windowSize']).loss
    
    
    if net_hyperparameters['bidir']:
        weights = [0.5, 0.5]
        losses = [image_loss_func, image_loss_func]
    else: 
        weights = [1]
        losses = [image_loss_func]


    # Prepare deformation loss
    losses += [vxm.losses.Grad(net_hyperparameters['loss_penalty'], loss_mult = net_hyperparameters['int_downsize']).loss]
    # losses += [vxm.losses.GradJacobian(net_hyperparameters['loss_penalty'], loss_mult = net_hyperparameters['int_downsize'], jacobian_weight=1.0)]
    # grad_jacobian_loss = GradJacobian(penalty='l2', loss_mult=None, jacobian_weight=1.0)
    weights += [net_hyperparameters['lambda']]

    # TRAIN -----------------------------------------------------------------------------------------
    tictoc.tic()
    epoch_values = []
    for epoch in range(net_hyperparameters['initial_epoch'], net_hyperparameters['epochs']):

        # save model checkpoint
        if epoch % 20 == 0:
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(net_hyperparameters['steps_per_epoch']):

            step_start_time = time.time()

            # Generate inputs (and true outputs) and convert them to tensors (with dimensions permuted)
            inputs, y_true, inmasks, y_true_outmasks = next(generator)
            concat_inputs = inputs

            for i in inmasks:
                concat_inputs.append(i)

            inputs = [torch.from_numpy(d.astype(np.float32)).to(device).permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 3, 1, 2) for d in y_true] # aod - for 2d
            inmasks = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 3, 1, 2) for d in inmasks] # aod - for 2d
            y_true_outmasks = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 3, 1, 2) for d in y_true_outmasks] # aod - for 2d
            concat_inputs = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 3, 1, 2) for d in concat_inputs] # aod - for 2d

            # Predict warped image and flow
            y_pred, ym_pred = model(*concat_inputs)

            if net_hyperparameters['image_loss'] == 'dual_ncc_tsnr' or net_hyperparameters['image_loss'] == 'dual_ncc_dice':
                # Calculate total loss
                loss = 0
                loss_list = []
                for n, loss_function in enumerate(losses):
                    if n == len(y_pred)-1:
                        curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    else:
                        curr_loss = loss_function(y_true[n], y_pred[n], y_true_outmasks[0], ym_pred[0]) * weights[n]

                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            elif net_hyperparameters['image_loss'] == 'single':
                loss = 0
                loss_list = []
                for n, loss_function in enumerate(losses):
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss
            
            if net_hyperparameters['jac_regularizer']: 
                jacobian_loss = -vxm.losses.negativeJac_loss(y_pred[-1])
                total_loss = loss + jacobian_loss
            else:
                total_loss = loss
            
            
            epoch_loss.append(loss_list)
            epoch_total_loss.append(total_loss.item())

            # Backpropagate and optimize
            optimizer.zero_grad() #clear current gradient values!
            loss.backward() #calculate gradients. When you call the backward() method on a tensor, it deallocates the intermediate buffers by default to save memory. If you need to compute gradients through the same graph multiple times, you should specify retain_graph=True to retain the graph's computational history.
            optimizer.step() #update parameters

            # Get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # Print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, net_hyperparameters['epochs'])
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        run["training/epoch/epoch_total_loss"].log(np.mean(epoch_total_loss))

        if math.isnan(np.mean(epoch_total_loss)):
            print("Epoch loss is NaN. Exiting the training loop.")
            return  # Exit the 'train' function
           # Check for early stopping
        
        ## Early stopping
        # training_loss = np.mean(epoch_total_loss)
        # # Check if the training loss has improved
        # if training_loss < best_training_loss:
        #     best_training_loss = training_loss
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1

        # if epochs_without_improvement >= early_stopping_patience:
        #     print("Early stopping: No improvement in training loss for", early_stopping_patience, "epochs.")
        #     break  # Exit the training loop

        epoch_values.append(np.mean(epoch_total_loss))

      
    # Final model save
    model.save(os.path.join(model_dir, '%04d.pt' % net_hyperparameters['epochs']))
    tictoc.toc()

    plt.plot(range(0, net_hyperparameters['epochs']), epoch_values)
    plt.savefig(model_dir + '/logs.png')
    plt.close()

    # Specify the CSV file path where you want to save the list
    import csv
    csv_file_path = model_dir + '/logs.csv'

    # Open the CSV file for writing
    with open(csv_file_path, mode="w", newline="") as file:
    # Create a CSV writer
        csv_writer = csv.writer(file, delimiter=",")

        # Write the list of values to the CSV file
        csv_writer.writerow(epoch_values)


if __name__ == '__main__': 
    
    # PATHS ------------------
    #####################################################
    all = True                                          
    imgtype = "Native"                                  #
    # imgtype = "Allograft"    
    with open('./data_paths_' + imgtype+ '.txt') as f:    
        studies = f.read().splitlines()                 
    main_path = 'D:/RM_RENAL/Registration/Voxelmorph/'
    main_model_dir = main_path + '/' + imgtype + '/Models_dualw/'
    #####################################################

    # MAIN PARAMETERS ------------------
    #############################################################################
    nstudies = 27
    print(nstudies)
    args = {    'imgType': imgtype, # or Allograft
                'regType': 'groupwise', # or groupwise
                'm0_reg': 0, #1:exclude M0
                'data_path': 'D:/RM_RENAL/Registration/Elastix/PCASL/' + imgtype + '/' + studies[nstudies] + 'MI/result.0.nii',
                'labels_path_cortex': 'D:/RM_RENAL/DATA/' + imgtype + '/masks/' + studies[nstudies] + 'cortex_3d.nii',
                'labels_path_kidney': 'D:/RM_RENAL/DATA/' + imgtype + '/masks/' + studies[nstudies] + 'kidney_3d_maskrcnn.nii'
        }

    trainData_path = main_path + args['imgType'] + '/train_data/'
    trainLabel_path = main_path + args['imgType'] + '/train_labels/'  
    
        
    # Experiments---------------------------------------------------------------------------------------------------
    lambda_values = [1, 2]
    window_size = 9

    # ## a) single loss function
    # # ----------------------------------------------------------------------------
    # for lambda_value in lambda_values:
    #     net_hyperparameters = { 'img_prefix': None, 
    #                                 'img_suffix': None, 
    #                                 'batch_size': 8, 
    #                                 'load_model': False, 
    #                                 'bidir': False, 
    #                                 'aug': False,
    #                                 'int_steps': 7,  
    #                                 'int_downsize': 1, 
    #                                 'lr': 1e-3, 
    #                                 'image_loss': 'single', 
    #                                 'loss_penalty': 'l2',
    #                                 'jac_regularizer': False,
    #                                 'lambda': lambda_value, 
    #                                 'initial_epoch': 0, 
    #                                 'epochs': 1000, 
    #                                 'steps_per_epoch': 100, 
    #                                 'resize': None, 
    #                                 'windowSize': [window_size, window_size]
    #                             }

    #     model_dir = main_model_dir + \
    #                     'ncc_' + \
    #                     'e' + str(net_hyperparameters['epochs']) + \
    #                     '_s' + str(net_hyperparameters['steps_per_epoch']) + \
    #                     '_lr' + str(net_hyperparameters['lr']) + \
    #                     '_l' + str(net_hyperparameters['lambda']) + \
    #                     '_bs' + str(net_hyperparameters['batch_size']) + \
    #                     '_jacobian_p' + net_hyperparameters['loss_penalty'] + \
    #                     '_ints' + str(net_hyperparameters['int_steps']) + \
    #                     '_ws' + str(net_hyperparameters['windowSize'][0]) + \
    #                     '_intdown' + str(net_hyperparameters['int_downsize']) 

    #     os.makedirs(model_dir, exist_ok=True)
    #     train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full')

    #     # Bidir
    #     net_hyperparameters = { 'img_prefix': None, 
    #                                 'img_suffix': None, 
    #                                 'batch_size': 8, 
    #                                 'load_model': False, 
    #                                 'bidir': True, 
    #                                 'aug': False,
    #                                 'int_steps': 7,  
    #                                 'int_downsize': 1, 
    #                                 'lr': 1e-3, 
    #                                 'image_loss': 'single', 
    #                                 'loss_penalty': 'l2',
    #                                 'jac_regularizer': False,
    #                                 'lambda': lambda_value, 
    #                                 'initial_epoch': 0, 
    #                                 'epochs': 1000, 
    #                                 'steps_per_epoch': 100, 
    #                                 'resize': None, 
    #                                 'windowSize': [window_size, window_size]
    #                             }

    #     model_dir = main_model_dir + \
    #                     'ncc_bidir_' + \
    #                     'e' + str(net_hyperparameters['epochs']) + \
    #                     '_s' + str(net_hyperparameters['steps_per_epoch']) + \
    #                     '_lr' + str(net_hyperparameters['lr']) + \
    #                     '_l' + str(net_hyperparameters['lambda']) + \
    #                     '_bs' + str(net_hyperparameters['batch_size']) + \
    #                     '_p' + net_hyperparameters['loss_penalty'] + \
    #                     '_ints' + str(net_hyperparameters['int_steps']) + \
    #                     '_ws' + str(net_hyperparameters['windowSize'][0]) + \
    #                     '_intdown' + str(net_hyperparameters['int_downsize']) 

    #     os.makedirs(model_dir, exist_ok=True)
    #     train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full') 
 
    ## b) ncc-dice loss function
    # Unir
    # ncc_weights = [0.9, 0.8]
    # dice_weights = [0.1, 0.2]
    # for lambda_value in lambda_values:
    #     for weights in range(1, 2):
    #         net_hyperparameters = { 'img_prefix': None, 
    #                                     'img_suffix': None, 
    #                                     'batch_size': 8, 
    #                                     'load_model': False, 
    #                                     'bidir': False, 
    #                                     'aug': False,
    #                                     'int_steps': 7,  
    #                                     'int_downsize': 1, 
    #                                     'lr': 1e-3, 
    #                                     'image_loss': 'dual_ncc_dice', 
    #                                     'ncc_weight' : ncc_weights[weights], 
    #                                     'dice_weight': dice_weights[weights], 
    #                                     'loss_penalty': 'l2',
    #                                     'jac_regularizer': False,
    #                                     'lambda': lambda_value, 
    #                                     'initial_epoch': 0, 
    #                                     'epochs': 1000, 
    #                                     'steps_per_epoch': 100, 
    #                                     'resize': None, 
    #                                     'windowSize': [window_size, window_size]
    #                                 }

    #         model_dir = main_model_dir + \
    #                         'ncc-dice_' + \
    #                         'e' + str(net_hyperparameters['epochs']) + \
    #                         '_s' + str(net_hyperparameters['steps_per_epoch']) + \
    #                         '_lr' + str(net_hyperparameters['lr']) + \
    #                         '_l' + str(net_hyperparameters['lambda']) + \
    #                         '_bs' + str(net_hyperparameters['batch_size']) + \
    #                         '_p' + net_hyperparameters['loss_penalty'] + \
    #                         '_ints' + str(net_hyperparameters['int_steps']) + \
    #                         '_ws' + str(net_hyperparameters['windowSize'][0]) + \
    #                         '_intdown' + str(net_hyperparameters['int_downsize']) + \
    #                         '_ncc' + str(net_hyperparameters['ncc_weight']) + \
    #                         '_dice' + str(net_hyperparameters['dice_weight'])

    #         os.makedirs(model_dir, exist_ok=True)
    #         train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full')

    #         # Bidir 
    #         net_hyperparameters = { 'img_prefix': None, 
    #                                     'img_suffix': None, 
    #                                     'batch_size': 8, 
    #                                     'load_model': False, 
    #                                     'bidir': True, 
    #                                     'aug': False,
    #                                     'int_steps': 7,  
    #                                     'int_downsize': 1, 
    #                                     'lr': 1e-3, 
    #                                     'image_loss': 'dual_ncc_dice', 
    #                                     'ncc_weight' : ncc_weights[weights], 
    #                                     'dice_weight': dice_weights[weights], 
    #                                     'loss_penalty': 'l2',
    #                                     'jac_regularizer': False,
    #                                     'lambda': lambda_value, 
    #                                     'initial_epoch': 0, 
    #                                     'epochs': 1000, 
    #                                     'steps_per_epoch': 100, 
    #                                     'resize': None, 
    #                                     'windowSize': [window_size, window_size]
    #                                 }

    #         model_dir = main_model_dir + \
    #                         'ncc-dice_bidir_' + \
    #                         'e' + str(net_hyperparameters['epochs']) + \
    #                         '_s' + str(net_hyperparameters['steps_per_epoch']) + \
    #                         '_lr' + str(net_hyperparameters['lr']) + \
    #                         '_l' + str(net_hyperparameters['lambda']) + \
    #                         '_bs' + str(net_hyperparameters['batch_size']) + \
    #                         '_p' + net_hyperparameters['loss_penalty'] + \
    #                         '_ints' + str(net_hyperparameters['int_steps']) + \
    #                         '_ws' + str(net_hyperparameters['windowSize'][0]) + \
    #                         '_intdown' + str(net_hyperparameters['int_downsize']) + \
    #                         '_ncc' + str(net_hyperparameters['ncc_weight']) + \
    #                         '_dice' + str(net_hyperparameters['dice_weight'])

    #         os.makedirs(model_dir, exist_ok=True)
    #         train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full') 

    ## c) ncc-tsnr loss function
    ncc_weights = [0.9, 0.95]
    tsnr_weights = [0.1, 0.05]
    for lambda_value in lambda_values:
        for weights in range(1, 2):
            # Unir
            net_hyperparameters = { 'img_prefix': None, 
                                        'img_suffix': None, 
                                        'batch_size': 8, 
                                        'load_model': False, 
                                        'bidir': False, 
                                        'aug': False,
                                        'int_steps': 7,  
                                        'int_downsize': 1, 
                                        'lr': 1e-3, 
                                        'image_loss': 'dual_ncc_tsnr', 
                                        'ncc_weight' : ncc_weights[weights], 
                                        'tsnr_weight': tsnr_weights[weights], 
                                        'loss_penalty': 'l2',
                                        'jac_regularizer': False,
                                        'lambda': lambda_value, 
                                        'initial_epoch': 0, 
                                        'epochs': 1000, 
                                        'steps_per_epoch': 100, 
                                        'resize': None, 
                                        'windowSize': [window_size, window_size]
                                    }

            model_dir = main_model_dir + \
                            'ncc-tsnr_' + \
                            'e' + str(net_hyperparameters['epochs']) + \
                            '_s' + str(net_hyperparameters['steps_per_epoch']) + \
                            '_lr' + str(net_hyperparameters['lr']) + \
                            '_l' + str(net_hyperparameters['lambda']) + \
                            '_bs' + str(net_hyperparameters['batch_size']) + \
                            '_p' + net_hyperparameters['loss_penalty'] + \
                            '_ints' + str(net_hyperparameters['int_steps']) + \
                            '_ws' + str(net_hyperparameters['windowSize'][0]) + \
                            '_intdown' + str(net_hyperparameters['int_downsize']) + \
                            '_ncc' + str(net_hyperparameters['ncc_weight']) + \
                            '_tsnr' + str(net_hyperparameters['tsnr_weight'])

            os.makedirs(model_dir, exist_ok=True)
            train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full')

            # Bidir
            net_hyperparameters = { 'img_prefix': None, 
                                        'img_suffix': None, 
                                        'batch_size': 8, 
                                        'load_model': False, 
                                        'bidir': True, 
                                        'aug': False,
                                        'int_steps': 7,  
                                        'int_downsize': 1, 
                                        'lr': 1e-3, 
                                        'image_loss': 'dual_ncc_tsnr', 
                                        'ncc_weight' : ncc_weights[weights], 
                                        'tsnr_weight': tsnr_weights[weights], 
                                        'loss_penalty': 'l2',
                                        'jac_regularizer': False,
                                        'lambda': lambda_value, 
                                        'initial_epoch': 0, 
                                        'epochs': 1000, 
                                        'steps_per_epoch': 100, 
                                        'resize': None, 
                                        'windowSize': [window_size, window_size]
                                    }

            model_dir = main_model_dir + \
                            'ncc-tsnr_bidir_' + \
                            'e' + str(net_hyperparameters['epochs']) + \
                            '_s' + str(net_hyperparameters['steps_per_epoch']) + \
                            '_lr' + str(net_hyperparameters['lr']) + \
                            '_l' + str(net_hyperparameters['lambda']) + \
                            '_bs' + str(net_hyperparameters['batch_size']) + \
                            '_p' + net_hyperparameters['loss_penalty'] + \
                            '_ints' + str(net_hyperparameters['int_steps']) + \
                            '_ws' + str(net_hyperparameters['windowSize'][0]) + \
                            '_intdown' + str(net_hyperparameters['int_downsize']) + \
                            '_ncc' + str(net_hyperparameters['ncc_weight']) + \
                            '_tsnr' + str(net_hyperparameters['tsnr_weight'])

            os.makedirs(model_dir, exist_ok=True)
            train(args, trainData_path, trainLabel_path, net_hyperparameters, model_dir, mode = 'full') 
    
    