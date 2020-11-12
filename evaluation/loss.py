import lpips
import torch
import pytorch_ssim
import numpy as np
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
ssim_loss = pytorch_ssim.SSIM(window_size = 11)

def lpips_dis( x, y):
    # x, y size : ( 3, N,N), cv2 readed image, value range (0,255)
    # change it to pytorch tensor, need to be normalized to [-1,1]
    x = torch.tensor( x, dtype=torch.float32).cuda()
    y = torch.tensor( y, dtype=torch.float32).cuda()
    x = (x/255.0 - 0.5) *2 
    x = torch.clamp(x, min=-1, max=1)
    x = x.unsequeeze(0)

    y = (y/255.0 - 0.5) *2 
    y = torch.clamp(y, min=-1, max=1)
    y = y.unsequeeze(0)
    d = loss_fn_alex(x, y)
    return d

def l2_dis(x, y):
    mse = np.mean( (x - y) ** 2 )
    return mse


def ssim_dis(x, y):
    # x, y size : ( 3, N,N), cv2 readed image, value range (0,255)
    x = torch.tensor( x, dtype=torch.float32).cuda().unsequeeze(0)
    y = torch.tensor( y, dtype=torch.float32).cuda().unsequeeze(0)
    d = ssim_loss( x, y)
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