import numpy as np 
import os 
import cv2
import loss
def get_list(method = 'deep3dR'):
    img_list = []
    if method == 'deep3dR':
        for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/data/data'):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    if 'deep3dfaceR' in img_path:
                        if '_no_tex.png' not in img_path:
                            print (img_path)
                            img_list.append(img_path)
    elif method == 'MGCNet':
        for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/data/data'):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    if 'mgcnet' in img_path:
                        if '_mulPoses' not in img_path:
                            print (img_path)
                            img_list.append(img_path)
    elif method =='ours':
        for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/data/data'):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    if 'ours' in img_path and 'ours_render' not in img_path:
                        if '_mulPoses' not in img_path:
                            print (img_path)
                            img_list.append(img_path)


    return img_path

def calculate(method = 'ours'):
    
    img_list = get_list(method)

    l2_loss = []
    ssim_loss = []
    lpips_loss = []
    cpbd_loss = []
    gt_cpbd_loss = []
    if method== 'ours':
        for img_p in img_list:
            result = cv2.imread(img_p)
            print (result.shape)
            high = result.shape[0]
            width = int(result.shape[1] /3)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
            l2_loss.append( loss.l2_dis(  rec_img , gt_img) )
            ssim_loss.append(loss.ssim_dis(  rec_img, gt_img ))
            lpips_loss.append(loss.lpips_dis(rec_img, gt_img))
            cpbd_loss.append( loss.cpbd(rec_img )  )
            gt_cpbd_loss.append( loss.cpbd(gt_img)  )
    
    print (l2_loss)
    print (ssim_loss)
    print (cpbd_loss)
    print (lpips_loss)
    print (gt_cpbd_loss)



calculate(method = 'ours')