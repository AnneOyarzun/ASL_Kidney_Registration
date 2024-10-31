import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt

def elastixRegistration(fix_to_register, mov_to_register, outPath, paramPath, maskPath = None):
    print('Registering...')
    if maskPath:
        comm = comm = 'elastix ' + '-f ' + fix_to_register + ' -m ' + mov_to_register + ' -out ' + outPath + ' -p ' + paramPath + ' -fMask ' + maskPath + ' -mMask ' + maskPath
    else:
        comm = 'elastix ' + '-f ' + fix_to_register + ' -m ' + mov_to_register + ' -out ' + outPath + ' -p ' + paramPath
    os.system(comm) 
    print('Registration completed') 

def transformixRegistration(path_to_binary_mask, path_to_transformation_parameters, output_directory):
    print('Registering...')
    #comm = 'transformix' + ' -in ' + path_to_binary_mask + ' -tp ' + path_to_transformation_parameters + ' -out ' + output_directory
    comm = 'transformix' + ' -in ' + path_to_binary_mask + ' -out ' + output_directory + ' -tp ' + path_to_transformation_parameters + ' -def ' + path_to_binary_mask + ' -nopad -fct 0'
    os.system(comm) 
    print('Registration completed') 

def transformixRegistration_jac(fix_to_register, path_to_transformation_parameters, output_directory):
    # Para transformacones como StackBSTransform no se puede computan el determinante
    print('Registering...')
    comm = 'transformix' + ' -in ' + fix_to_register + ' -out ' + output_directory + ' -tp ' + path_to_transformation_parameters + ' -def ' + fix_to_register +  ' -jac all -target ' + output_directory
    os.system(comm) 
    print('Registration completed') 

def read_parameter_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def save_parameter_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

def modify_param_path_binary(param_path, metric): 
    '''
    To modify interpolation options of the elastix transformation parameter map. 
    '''
    parameter_file = read_parameter_file(param_path + 'TransformParameters.0.txt')
    parameter_file_mod = parameter_file
    if metric == 'PCA2' or metric == 'PCA2_BSplineTransform':
        parameter_file_mod[29] = '(ResampleInterpolator "FinalNearestNeighborInterpolator")\n'
        parameter_file_mod[30] = '(FinalBSplineInterpolationOrder 0)\n'
        save_parameter_file(param_path + 'Binary_TransformParameters.0.txt', parameter_file_mod)
    elif metric == 'MI': 
        parameter_file_mod[23] = '(ResampleInterpolator "FinalNearestNeighborInterpolator")\n'
        parameter_file_mod[24] = '(FinalBSplineInterpolationOrder 0)\n'
        save_parameter_file(param_path + 'Binary_TransformParameters.0.txt', parameter_file_mod)
    elif metric == 'NCC': 
        parameter_file_mod[29] = '(ResampleInterpolator "FinalNearestNeighborInterpolator")\n'
        parameter_file_mod[30] = '(FinalBSplineInterpolationOrder 0)\n'
        save_parameter_file(param_path + 'Binary_TransformParameters.0.txt', parameter_file_mod)
    else:
        print('Metric not defined')
    


if __name__ == '__main__': 
                                       
    pre_norm = False
    compute_registration = False
    compute_transformix = True
    compute_transformix_jac = True
    # mask_option = 'with_masks'
    # mask_option = 'without_masks'
    metric = "NCC"
    # metric = 'PCA2_BSplineTransform'
    paramPath_PCA2 = 'Z:/RM_RENAL/Registration/parameter_maps/pca2_groupwise_kidney_asl_bsplinetransform.txt'



    # Elastix registration (ASL images
    # ---------------------------------------------------------------------------------------------------------
    imgtype = "Allograft"  
    
    with open('./data_paths_' + imgtype+ '.txt') as f:  #  
        studies = f.read().splitlines() 
 
    
    # Paths
    main_path = 'Z:/RM_RENAL/Registration/'
    unregdata_path = main_path + 'Unregistered_Data/' + imgtype + '/'
    reg_path = 'Z:/RM_RENAL/Registration/Elastix/PCASL/'
    data_path = 'Z:/RM_RENAL/DATA/'

    for nstudies in range(0, len(studies)):
        mhdPath = unregdata_path +  studies[nstudies] + 'imgs_to_register.mhd'
        mhdfiles = sitk.ReadImage(mhdPath)

        if pre_norm:          
            mhdfiles = sitk.ReadImage(mhdPath)
            for i in range(0, mhdfiles.GetSize()[2]):
                mhdfiles[:,:,i] = sitk.RescaleIntensity(mhdfiles[:,:,i], 0, 255)
            sitk.WriteImage(mhdfiles, unregdata_path +  studies[nstudies] + 'imgs_to_register_res.mhd')
            mhdPath = unregdata_path +  studies[nstudies] + 'imgs_to_register_res.mhd'

        # Elastix
        if compute_registration:
            if not os.path.exists(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/'): 
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/')
            # maskPath = data_path + imgtype + '/masks/' + studies[nstudies] + 'kidney_3d_maskrcnn.nii'
            maskPath = None
            if maskPath:
                output_directory_PCA2 = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/'
                os.makedirs(output_directory_PCA2, exist_ok=True)
            else:
                output_directory_PCA2 = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/'
                os.makedirs(output_directory_PCA2, exist_ok=True)

            elastixRegistration(mhdPath, mhdPath, output_directory_PCA2, paramPath_PCA2, maskPath=maskPath)   

        if compute_transformix_jac:
            if not os.path.exists(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/transformix/'): 
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/transformix/')
            output_directory = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/transformix/'
            path_to_transformation_parameters = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/TransformParameters.0.txt'

            transformixRegistration_jac(mhdPath, path_to_transformation_parameters, output_directory)

        # Transformix (for masks)
        if compute_transformix:
            path_to_binary_cortex = data_path + imgtype + '/masks/' + studies[nstudies] + 'cortex_3d.nii'            
            path_to_binary_kidney = data_path + imgtype + '/masks/' + studies[nstudies] + 'kidney_3d_maskrcnn.nii'    

            # Check if origin/spacing/direction corresponds to the images
            cortex = sitk.ReadImage(path_to_binary_cortex)
            if cortex.GetOrigin() != mhdfiles.GetOrigin() or cortex.GetSpacing() != mhdfiles.GetSpacing() or cortex.GetDirection() != mhdfiles.GetDirection():
                cortex.SetOrigin(mhdfiles.GetOrigin())
                cortex.SetSpacing(mhdfiles.GetSpacing())
                cortex.SetDirection(mhdfiles.GetDirection())
                sitk.WriteImage(cortex, path_to_binary_cortex)                
            kidney = sitk.ReadImage(path_to_binary_kidney)
            if kidney.GetOrigin() != mhdfiles.GetOrigin() or kidney.GetSpacing() != mhdfiles.GetSpacing() or kidney.GetDirection() != mhdfiles.GetDirection():
                kidney.SetOrigin(mhdfiles.GetOrigin())
                kidney.SetSpacing(mhdfiles.GetSpacing())
                kidney.SetDirection(mhdfiles.GetDirection())
                sitk.WriteImage(kidney, path_to_binary_kidney)

            path_to_transformation_parameters = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/'
            modify_param_path_binary(path_to_transformation_parameters, metric)

            if not os.path.exists(reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/'): 
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Cortex/')
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Kidney')

            output_directory_cortex = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Cortex/'
            output_directory_kidney = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Kidney/'
            
            transformixRegistration(path_to_binary_kidney, path_to_transformation_parameters + 'Binary_TransformParameters.0.txt', output_directory_kidney)
            transformixRegistration(path_to_binary_cortex, path_to_transformation_parameters + 'Binary_TransformParameters.0.txt', output_directory_cortex)
   

    # #################################################################
    # ################################################################
    imgtype = "Native"  
    
    with open('./data_paths_' + imgtype+ '.txt') as f:  #  
        studies = f.read().splitlines() 
 
    
    # Paths
    main_path = 'Z:/RM_RENAL/Registration/'
    unregdata_path = main_path + 'Unregistered_Data/' + imgtype + '/'
    reg_path = 'Z:/RM_RENAL/Registration/Elastix/PCASL/'
    paramPath_PCA2 = 'Z:/RM_RENAL/Registration/parameter_maps/pca2_groupwise_kidney_asl_bsplinetransform_ncc.txt'
    data_path = 'Z:/RM_RENAL/DATA/'

    for nstudies in range(0, len(studies)):
        mhdPath = unregdata_path +  studies[nstudies] + 'imgs_to_register.mhd'
        mhdfiles = sitk.ReadImage(mhdPath)

        if pre_norm:          
            mhdfiles = sitk.ReadImage(mhdPath)
            for i in range(0, mhdfiles.GetSize()[2]):
                mhdfiles[:,:,i] = sitk.RescaleIntensity(mhdfiles[:,:,i], 0, 255)
            sitk.WriteImage(mhdfiles, unregdata_path +  studies[nstudies] + 'imgs_to_register_res.mhd')
            mhdPath = unregdata_path +  studies[nstudies] + 'imgs_to_register_res.mhd'

        # Elastix
        if compute_registration:
            if not os.path.exists(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/'): 
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/')
            # maskPath = data_path + imgtype + '/masks/' + studies[nstudies] + 'kidney_3d_maskrcnn.nii'
            maskPath = None
            if maskPath:
                output_directory_PCA2 = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/'
                os.makedirs(output_directory_PCA2, exist_ok=True)
            else:
                output_directory_PCA2 = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/'
                os.makedirs(output_directory_PCA2, exist_ok=True)

            elastixRegistration(mhdPath, mhdPath, output_directory_PCA2, paramPath_PCA2, maskPath=maskPath)   

        if compute_transformix_jac:
            if not os.path.exists(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/transformix/'): 
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + '/' + metric + '/' + mask_option + '/transformix/')
            output_directory = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/transformix/'
            path_to_transformation_parameters = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/TransformParameters.0.txt'

            transformixRegistration_jac(mhdPath, path_to_transformation_parameters, output_directory)

        # Transformix (for masks)
        if compute_transformix:
            path_to_binary_cortex = data_path + imgtype + '/masks/' + studies[nstudies] + 'cortex_3d.nii'            
            path_to_binary_kidney = data_path + imgtype + '/masks/' + studies[nstudies] + 'kidney_3d_maskrcnn.nii'    

            # Check if origin/spacing/direction corresponds to the images
            cortex = sitk.ReadImage(path_to_binary_cortex)
            if cortex.GetOrigin() != mhdfiles.GetOrigin() or cortex.GetSpacing() != mhdfiles.GetSpacing() or cortex.GetDirection() != mhdfiles.GetDirection():
                cortex.SetOrigin(mhdfiles.GetOrigin())
                cortex.SetSpacing(mhdfiles.GetSpacing())
                cortex.SetDirection(mhdfiles.GetDirection())
                sitk.WriteImage(cortex, path_to_binary_cortex)                
            kidney = sitk.ReadImage(path_to_binary_kidney)
            if kidney.GetOrigin() != mhdfiles.GetOrigin() or kidney.GetSpacing() != mhdfiles.GetSpacing() or kidney.GetDirection() != mhdfiles.GetDirection():
                kidney.SetOrigin(mhdfiles.GetOrigin())
                kidney.SetSpacing(mhdfiles.GetSpacing())
                kidney.SetDirection(mhdfiles.GetDirection())
                sitk.WriteImage(kidney, path_to_binary_kidney)

            path_to_transformation_parameters = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/'
            modify_param_path_binary(path_to_transformation_parameters, metric)

            if not os.path.exists(reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/'): 
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Cortex/')
                os.makedirs(reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Kidney')

            output_directory_cortex = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Cortex/'
            output_directory_kidney = reg_path + imgtype + '/' + studies[nstudies] + metric + '/' + mask_option + '/masks/Kidney/'
            
            transformixRegistration(path_to_binary_kidney, path_to_transformation_parameters + 'Binary_TransformParameters.0.txt', output_directory_kidney)
            transformixRegistration(path_to_binary_cortex, path_to_transformation_parameters + 'Binary_TransformParameters.0.txt', output_directory_cortex)
   