import torch
import torch.nn.functional as F
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
import pytorch_ssim



class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self,  y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class TSNR:
    """
    Temporal signal-to-noise ratio.
    """
    def loss(self, y_true, y_pred, ym_true, ym_pred):
        ytrue_mean_list = []
        ypred_mean_list = []

        mask_true = ym_true.bool() # Convert Tensor into BoolTensor
        mask_pred = ym_pred.bool()

        for i in range(len(y_true)): 
            It = torch.mean(torch.masked_select(y_true[i], mask_true[i]))
            Ip = torch.mean(torch.masked_select(y_pred[i], mask_pred[i]))

            ytrue_mean_list.append(It)
            ypred_mean_list.append(Ip)

        tsnr_y_true = torch.mean(torch.stack(ytrue_mean_list, dim = 0)) / torch.std(torch.stack(ytrue_mean_list, dim = 0)) 
        tsnr_y_pred = torch.mean(torch.stack(ypred_mean_list, dim = 0)) / torch.std(torch.stack(ypred_mean_list, dim = 0))
  
        tsnr_incr = tsnr_y_pred/tsnr_y_true
        
        return -tsnr_incr
    
class Duo_NCC_TSNR:
    def __init__(self, win = None, ncc_weight=0.99, tsnr_weight=0.01):
        self.win = win
        self.ncc_weigth = ncc_weight
        self.tsnr_weigth = tsnr_weight

    def loss(self, y_true, y_pred, ym_true, ym_pred):
        
        ncc_loss = NCC(win=self.win).loss(y_true, y_pred) 
        tsnr_loss = TSNR().loss(y_true, y_pred, ym_true, ym_pred) 
        tsnr_loss_normalized = tsnr_loss / (4 - tsnr_loss)
        total_loss = ncc_loss * self.ncc_weigth + tsnr_loss_normalized * self.tsnr_weigth

        return total_loss
    


class Duo_NCC_DICE:
    def __init__(self, win = None, ncc_weight=0.8, dice_weight=0.2):
        self.win = win
        self.ncc_weight = ncc_weight
        self.dice_weight = dice_weight
    
    def loss(self, y_true, y_pred, ym_true, ym_pred):
        # Calculate NCC loss
        ncc_loss = NCC(win=self.win).loss(y_true, y_pred) 
        

        # Calculate Dice coefficient loss
        dice_loss = Dice().loss(ym_true, ym_pred) 

        # Combine the losses with specified weights
        total_loss = (
            self.ncc_weight * ncc_loss +
            self.dice_weight * dice_loss
        )

        return total_loss


# class Dice:
#     """
#     N-D dice for segmentation
#     """

#     def loss(self, y_true, y_pred):
#         ndims = len(list(y_pred.size())) - 2
#         vol_axes = list(range(2, ndims + 2))
#         top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#         bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
#         dice = torch.mean(top / bottom)

#         # return -dice. Uso en negativo para entrenar solo con ncc o con dual_ncc_tsnr
#         return -dice

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
            raise ValueError("Both y_true and y_pred must be PyTorch tensors")

        ndims = len(y_pred.size()) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)

        # return -dice. Uso en negativo para entrenar solo con ncc o con dual_ncc_tsnr
        return -dice


class Grad:
    """
    N-D gradient loss. In 2D (Modified from the original one)
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy) 

        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def calculate_jacobian_determinant(flow): 

    '''
    adapted from: https://github.com/dykuang/Medical-image-registration/blob/master/source/losses.py
    Calculate the Jacobian value at each point of the displacement map having

    size of b*h*w*d*2

    The function calculates finite differences in the x and y directions within the displacement field. 
    It computes the differences between neighboring elements in the x and y dimensions. 
    This results in two 3D tensors, D_x and D_y, where the dimensions are [batch_size, num_channels, height - 1, width - 1].

    '''
    D_y = flow[:, 1:, :-1, :] - flow[:, :-1, :-1, :]
    D_x = flow[:, :-1, 1:, :] - flow[:, :-1, :-1, :]

    D1 = D_x[..., 0] * D_y[..., 1] - D_x[..., 1] * D_y[..., 0]

    return D1

def negativeJac_loss(ypred):
    """
    Penalizing locations where Jacobian has negative determinants.
    """
    neg_jac = 0.5 * (torch.abs(calculate_jacobian_determinant(ypred)) - calculate_jacobian_determinant(ypred))
    
    return torch.sum(neg_jac)



class MutualInformation():
    """
    Soft Mutual Information approximation for intensity volumes
    More information/citation:
    - Courtney K Guo. 
      Multi-modal image registration with unsupervised deep learning. 
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879
    """

    def loss(self, y_true, y_pred):
        return -self.volumes(y_true, y_pred)


class SSIM_Loss:
    def __init__(self, win=None):
        self.win = win
    def loss(self, y_true, y_pred):
        """Structural dissimilarity loss + L1 loss
        DSSIM is defined as (1-SSIM)/2
        https://en.wikipedia.org/wiki/Structural_similarity
        :param tensor y_true: Labeled ground truth
        :param tensor y_pred: Predicted labels, potentially non-binary
        :return float: 0.8 * DSSIM + 0.2 * L1
        """
        # mae = mean_absolute_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        # mae = torch.mean((y_true - y_pred) ** 2)
        # return 0.8 * (1.0 - ssim(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()) / 2.0) + 0.2 * mae
        ssim_loss = pytorch_ssim.SSIM(window_size = 7)

        return 1 - ssim_loss(y_true, y_pred)


def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field. 
    
    Parameters
    ----------
    disp : (n, 2, h, w) or (n, 3, d, h, w)
        displacement field
        
    image : (n, 1, d, h, w) or (1, 1, d, h, w)

    ref: https://github.com/vincentme/GroupRegNet/blob/master/model/loss.py
    '''

    image_shape = disp.shape
    dim = len(image_shape[2:])
    
    d_disp = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype = disp.dtype, device = disp.device)
    d_image = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype = disp.dtype, device = disp.device)
    
    # forward difference
    if dim == 2:
        d_disp[:, 1, :, :-1, :] = (disp[:, :, 1:, :] - disp[:, :, :-1, :])
        d_disp[:, 0, :, :, :-1] = (disp[:, :, :, 1:] - disp[:, :, :, :-1])
        d_image[:, 1, :, :-1, :] = (image[:, :, 1:, :] - image[:, :, :-1, :])
        d_image[:, 0, :, :, :-1] = (image[:, :, :, 1:] - image[:, :, :, :-1])
        
    elif dim == 3:
        d_disp[:, 2, :, :-1, :, :] = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        d_disp[:, 1, :, :, :-1, :] = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        d_disp[:, 0, :, :, :, :-1] = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])
        
        d_image[:, 2, :, :-1, :, :] = (image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        d_image[:, 1, :, :, :-1, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        d_image[:, 0, :, :, :, :-1] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1])

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim = 2, keepdims = True)*torch.exp(-torch.abs(d_image)))
    
    return loss  
        