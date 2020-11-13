import face_model
import argparse
import cv2
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
import pdb
import os
parser = argparse.ArgumentParser(description='face model test')
# general

parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model-0000', help='path to load model.')
parser.add_argument('--ga_model', default='', help='path to load model.')
parser.add_argument('--path', default='/u/lchen63/github/pix2pixHD/results/base1_no_pixel_att/test_latest/images', type=str, help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--method', default='ours', type=str, help='ver dist threshold')
args = parser.parse_args()

# model = face_model.FaceModel(args)
# print ("=========")

# img = cv2.imread('00001_real_image.jpg')
# img = cv2.resize(img,(112,112))

# img = model.get_input(img)
# f1 = model.get_feature(img)
# print(f1[0:10])
# print ('+++++++++++++++++')

# img = cv2.imread('00001_reference1.jpg')
# # img = cv2.imread('00001_reference1.jpg')

# img = cv2.resize(img,(112,112))
# img = model.get_input(img)
# f2 = model.get_feature(img)
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)

def read_videos( video_path, crop= False, pos=0):
    cap = cv2.VideoCapture(str(video_path))
    real_video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop:
                widths = frame.shape[1]
                print (frame.shape)
                ####nips#####
                # w = int(widths / 6)
                ####self#####
                w = int(widths / 8)
                # assert w == 256
                frame = frame[:, pos*w:pos*w+w,:]
                ####few-shot######
                # w = int(widths / 3)
                # assert w == 256
                # frame = frame[:, pos*w:pos*w+w,:]
                ####X2face########
                # w = int(widths / 2)
                # assert w == 256
                # frame = frame[:, :w,:]

                print (frame.shape)
            frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
            assert frame.shape[0] == 256
            real_video.append(frame)
        else:
            break

    return real_video


args = parser.parse_args()

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
    elif method =='ours-sh':
        for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/ours-sh'):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    
                    img_list.append(img_path)
    elif method =='jae':
        for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/rendering_tools/evaluation/fid_folder/jae'):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    
                    img_list.append(img_path)

    return img_list

def compute_CSIM(args, method ='ours'):
    # path = pathlib.Path(path)
    f_files = get_list(method)
    model = face_model.FaceModel(args)
    sims = []
    for i,d in enumerate(tqdm(f_files)):

        img_p = d

        if method =='ours':
            result = cv2.imread(img_p)
            high = result.shape[0]
            width = int(result.shape[1] /3)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
        elif method =='MGCNet':
            result = cv2.imread(img_p)
            high = result.shape[0]
            width = int(result.shape[1] /6)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
        elif method =='deep3dR':
            result = cv2.imread(img_p)
            print (result.shape)
            high = result.shape[0]
            width = int(result.shape[1] /2)

            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
        elif method =='ours-sh':
            result = cv2.imread(img_p)
            high = result.shape[0]
            width = int(result.shape[1] /2)
            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
        elif method =='jae':
            result = cv2.imread(img_p)
            high = result.shape[0]
            width = int(result.shape[1] /3)
            rec_img = result[:,width : width * 2, :]
            gt_img = result[:,:width,:]
        try:
            # rec_img = cv2.resize(rec_img,(112,112))
            print (rec_img.shape, gt_img.shape)
            f_i = model.get_input(rec_img)
            print (f_i.shape,'f_i')
            f1 = model.get_feature(f_i)
        
            # gt_img = cv2.resize(gt_img,(112,112))
            print (rec_img.shape, gt_img.shape)
            r_i = model.get_input(gt_img)
            print (r_i.shape, 'r_i')
            f2 = model.get_feature(r_i)
            dist = np.sum(np.square(f1-f2))
            sim = np.dot(f1, f2.T)
            sims.append(sim)
    
        except:
            continue

    print (sum(sims) / len(sims))

        

# path = '/home/cxu-serve/p1/common/experiment/nips_lrs/lrs_32_shot'
method = args.method
compute_CSIM(args,method)

# path = pathlib.Path(paths[0]) 0.603
# realfiles = list(path.glob('*real*.mp4'))
# realfiles.sort()

# path = pathlib.Path(paths[1])
# # fakefiles = list(path.glob('*fake*.mp4'))
# fakefiles = list(path.glob(os.path.join('*','test.mp4')))
# fakefiles.sort()


# dis_txt = open( os.path.join( os.path.dirname(path),  'ssim.txt')  ,'w')
# ssims = []
# psnrs =[]
# for i  in range(len(realfiles)):
#     if i == 5:
#         break

#     reals = read_videos(str(fakefiles[i]), crop=True, pos=1)
#     # try:
   
#     fakes = read_videos(str(fakefiles[i]), crop=True, pos=2)
#     # cv2.imwrite('/home/cxu-serve/p1/common/WSOL/tmp/real_{}.jpg'.format(0), reals[0])
#     # cv2.imwrite('/home/cxu-serve/p1/common/WSOL/tmp/false_{}.jpg'.format(0), fakes[0])

#     small = min(len(reals),len(fakes))

#     for  gg in range(small):
#         f_i = fakes[gg]
#         r_i = reals[gg]

#         assert f_i.shape[0] == 256
#         assert r_i.shape[0] == 256