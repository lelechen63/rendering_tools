import lpips
import torch
import numpy as np
import lpips

loss_fn_alex = lpips.LPIPS(net='alex')


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

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
    print ('!!!!!!')
    d = loss_fn_alex(x, y)
    return d

def l2_dis(x, y):
    mse = np.mean( (x - y) ** 2 )
    return mse


def ssim_dis(x, y):
    # x, y size : ( 3, N,N), cv2 readed image, value range (0,255)

    d = ssim( x, y)
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