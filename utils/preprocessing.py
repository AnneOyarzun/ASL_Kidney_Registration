import SimpleITK as sitk
import numpy as np
import cv2 as cv

def rescale(image, min=0, max=255):
    """ Rescale image intensity between 0 and 255 (default)
    :param image: image to rescale (simple itk image) :return: rescaled image """
    px_type = image.GetPixelIDTypeAsString()
    #print(px_type)
    if 'int' in px_type:
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        image = castImageFilter.Execute(image)

    out_image_rescaled = sitk.RescaleIntensity(image, min, max)
    return out_image_rescaled

def adjust_window_level( image, windowMin = 0, windowMax = 255):
    imageWL = sitk.IntensityWindowing(image,
                            windowMinimum=windowMin,
                            windowMaximum=windowMax)
    return imageWL

def fill_holes(label):
    '''
    It fills the holes (for example, for abdominal aorta, it fills the lumen hole that appears in the mask)
    :param label: label where the holes will be removed (simple itk)
    :return: label with removed holes
    '''

    fillHole=sitk.GrayscaleFillholeImageFilter

    if label.GetDimension()==3:
        x, y, z= label.GetSize()
        np_label = sitk.GetArrayFromImage(label)

        for slice in range(z):

            #para asignar un conjunto de filas y columnas hay que utilizar numpy
            noHole = sitk.GrayscaleFillhole(label[:, :, slice])

            np_label[slice, :, :] = sitk.GetArrayFromImage(noHole)

    elif label.GetDimension()==2:
        noHole = sitk.GrayscaleFillhole(label)
        np_label = sitk.GetArrayFromImage(noHole)

    withoutHoles=sitk.GetImageFromArray(np_label)
    withoutHoles.SetOrigin(label.GetOrigin())
    withoutHoles.SetSpacing(label.GetSpacing())

    return withoutHoles


def resize(image, new_size, label=None):
    '''
    Resize image
    :param image: image to resize (simple itk image)
    :param label: label to resize (optional) (usually necessary)
    :param new_size: new image size, as a list (specify size for each dimension eg, for 2d image: [224, 160], for 3d: [224, 160, 20] )
    :return: image resized, label resized (if provided as input)
    '''
    new_size = [int(i) for i in new_size]
    reference_image = sitk.Image(new_size, image.GetPixelIDValue())
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    reference_image.SetSpacing(
        [sz * spc / nsz for nsz, sz, spc in zip(new_size, image.GetSize(), image.GetSpacing())])

    interpolator = sitk.sitkLinear
    identity = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    interpolator_label = sitk.sitkNearestNeighbor
    out_image_resized = sitk.Resample(image, reference_image, identity, interpolator, 0, sitk.sitkFloat32)
    out_label_resized = None
    if label != None:
        out_label_resized = sitk.Resample(label, reference_image, identity, interpolator_label, 0,

                                          sitk.sitkUInt8)
    # rescale intensities
    return out_image_resized, out_label_resized

def specific_intensity_window(image, window_percent=0.2):
        image = sitk.Cast(image, sitk.sitkInt64)
        arr = sitk.GetArrayViewFromImage(image)
        min_val = arr.min()
        number_of_bins = arr.max() - min_val + 1

        hist = np.bincount((arr - min_val).ravel(), minlength=number_of_bins)
        hist_new = hist[1:]
        total = np.sum(hist_new)
        window_low = window_percent * total
        window_high = (1 - window_percent) * total
        cdf = np.cumsum(hist_new)
        low_intense = np.where(cdf >= window_low) + min_val
        high_intense = np.where(cdf >= window_high) + min_val
        res = sitk.IntensityWindowing(sitk.Cast(image, sitk.sitkFloat32), np.double(low_intense[0][0]),
                                      np.double(high_intense[0][0]), np.double(arr.min()), np.double(arr.max()))
        return res

def adjust_window_level( image, windowMin = 0, windowMax = 255):
    imageWL = sitk.IntensityWindowing(image,
                            windowMinimum=windowMin,
                            windowMaximum=windowMax)
    return imageWL

def image_padding(img, lower_bound, upper_bound, constant):
    filt = sitk.ConstantPadImageFilter()
    filt.SetConstant(constant)
    filt.SetPadLowerBound(lower_bound)
    filt.SetPadUpperBound(upper_bound)
    padded_img = filt.Execute(img)
    return padded_img


def hist_matching(im_orig, im_ref):
    # Pre-processing step (rescaling intensity)
    im_orig = rescale(im_orig, min = 0, max = 1)
    im_ref = rescale(im_ref, min = 0, max = 1)
    # Histogram matching
    histogram_match = sitk.HistogramMatchingImageFilter()
    histogram_match.SetThresholdAtMeanIntensity(True)  # useful if a lot of background pixels
    im_matched = histogram_match.Execute(im_orig, im_ref)
    return im_matched

def top_hat(img):
    filterSize =(25, 25)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
    tophat_img = cv.morphologyEx(sitk.GetArrayFromImage(img), cv.MORPH_TOPHAT, kernel)
    return tophat_img