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
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    if 'mgcnet' in img_path:
                        if '_mulPoses' in img_path:
                            print (img_path)
                            img_list.append(img_path)
    elif method =='ours':
        for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/data/data'):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    if 'ours' in img_path and 'ours_render' not in img_path:
                        print (img_path)
                        img_list.append(img_path)


    return img_list

def calculate(method = 'ours'):
    print (method)
    img_list = get_list(method)
    print (len(img_list))
    l2_loss = []
    ssim_loss = []
    lpips_loss = []
    cpbd_loss = []
    gt_cpbd_loss = []
    if method== 'ours':
        count = 0 
        for img_p in img_list:
            print ('+++')

            print (img_p)
            result = cv2.imread(img_p)
            print (result.shape)
            high = result.shape[0]
            width = int(result.shape[1] /3)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
            cv2.imwrite('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/ours/fake/%d.png'%count, rec_img)
            cv2.imwrite('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/ours/real/%d.png'%count, gt_img)
            l2_loss.append( loss.l2_dis(  rec_img , gt_img) )
            ssim_loss.append(loss.ssim_dis(  rec_img, gt_img ))
            lpips_loss.append(loss.lpips_dis(rec_img, gt_img))
            cpbd_loss.append( loss.cpbd_dis(rec_img )  )
            gt_cpbd_loss.append( loss.cpbd_dis(gt_img)  )
    elif method== 'MGCNet':
        for img_p in img_list:
            print ('+++')
            print (img_p)
            result = cv2.imread(img_p)
            print (result.shape)
            high = result.shape[0]
            width = int(result.shape[1] /6)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
            cv2.imwrite('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/MGCNet/fake/%d.png'%count, rec_img)
            cv2.imwrite('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/MGCNet/real/%d.png'%count, gt_img)

            l2_loss.append( loss.l2_dis(  rec_img , gt_img) )
            ssim_loss.append(loss.ssim_dis(  rec_img, gt_img ))
            lpips_loss.append(loss.lpips_dis(rec_img, gt_img))
            cpbd_loss.append( loss.cpbd_dis(rec_img )  )
            gt_cpbd_loss.append( loss.cpbd_dis(gt_img)  )

    elif method== 'deep3dR':
        for img_p in img_list:
            print (img_p)
            result = cv2.imread(img_p)
            print (result.shape)
            high = result.shape[0]
            width = int(result.shape[1] /2)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
            cv2.imwrite('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/deep3dR/fake/%d.png'%count, rec_img)
            cv2.imwrite('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/deep3dR/real/%d.png'%count, gt_img)

            l2_loss.append( loss.l2_dis(  rec_img , gt_img) )
            ssim_loss.append(loss.ssim_dis(  rec_img, gt_img ))
            lpips_loss.append(loss.lpips_dis(rec_img, gt_img))
            cpbd_loss.append( loss.cpbd_dis(rec_img )  )
            gt_cpbd_loss.append( loss.cpbd_dis(gt_img)  )

    print (l2_loss)
    print ('====================')
    print ('l2:', sum(l2_loss)/len(l2_loss))
    print (ssim_loss)
    print ('====================')
    print ('ssim_loss:' ,sum(ssim_loss)/len(ssim_loss))
    print (cpbd_loss)
    print ('====================')
    print ('cpbd_loss:' ,sum(cpbd_loss)/len(cpbd_loss))
    print (lpips_loss)
    print ('====================')
    print ('lpips_loss:' ,sum(lpips_loss)/len(lpips_loss))
    print (gt_cpbd_loss)
    print ('====================')
    print ('gt_cpbd_loss:' ,sum(gt_cpbd_loss)/len(gt_cpbd_loss))




calculate(method = 'ours')
calculate(method = 'MGCNet')
calculate(method = 'deep3dR')