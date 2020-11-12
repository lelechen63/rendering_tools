import lpips
import torch
import numpy as np
import lpips
import sys
from scipy import signal
from scipy import ndimage

from skimage.measure import compare_ssim

loss_fn_alex = lpips.LPIPS(net='alex')




def lpips_dis( x, y):
    print ('++++')
    # x, y size : ( N,N,3), cv2 readed image, value range (0,255)
    # change it to pytorch tensor, need to be normalized to [-1,1]
    x = torch.tensor( x, dtype=torch.float32)
    y = torch.tensor( y, dtype=torch.float32)
    print ('++++')
    x = (x/255.0 - 0.5) *2 
    print ('++++')
    x = torch.clamp(x, min=-1, max=1)
    print ('++++')
    x = x.permute(2,0,1).unsqueeze(0)
    print ('++++')

    y = (y/255.0 - 0.5) *2 
    y = torch.clamp(y, min=-1, max=1)
    print ('++++')
    y = y.permute(2,0,1).unsqueeze(0)

    print ('!!!!!!')
    d = loss_fn_alex(x, y)
    return d

def l2_dis(x, y):
    mse = np.mean( (x - y) ** 2 )
    return mse


def ssim_dis(x, y):
    # x, y size : ( 3, N,N), cv2 readed image, value range (0,255)

    d = compare_ssim( x, y, multichannel = True )
    return d


def psnr_f(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_dis(x, y):
    # x, y size : ( 3, N,N), cv2 readed image, value range (0,255)
    gray_x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    gray_y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    d = psnr_f(gray_x, gray_y)
    return d



def cpbd_dis(x):
    # xsize : ( 3, N,N), cv2 readed image, value range (0,255)
    d =  cpbd.compute(x)
    return d