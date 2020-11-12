import lpips
import torch
import numpy as np
import lpips
import sys
from scipy import signal
from scipy import ndimage
import cpbd
import cv2
from skimage.measure import compare_ssim

loss_fn_alex = lpips.LPIPS(net='alex')




def lpips_dis( x, y):
    # x, y size : ( N,N,3), cv2 readed image, value range (0,255)
    # change it to pytorch tensor, need to be normalized to [-1,1]
    x = torch.tensor( x, dtype=torch.float32)
    y = torch.tensor( y, dtype=torch.float32)
    x = (x/255.0 - 0.5) *2 
    x = torch.clamp(x, min=-1, max=1)
    x = x.permute(2,0,1).unsqueeze(0)

    y = (y/255.0 - 0.5) *2 
    y = torch.clamp(y, min=-1, max=1)
    y = y.permute(2,0,1).unsqueeze(0)

    d = loss_fn_alex(x, y)
    print (d.shape)
    return d.detach().numpy()

def l2_dis(x, y):
    mse = np.mean( (x - y) ** 2 )
    return mse


def ssim_dis(x, y):
    # x, y size : ( 3, N,N), cv2 readed image, value range (0,255)

    d = compare_ssim( x, y, multichannel = True )
    return d


def cpbd_dis(x):
    # xsize : (  N,N, 3), cv2 readed image, value range (0,255)
    input_image1 = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    d =  cpbd.compute(input_image1)
    return d