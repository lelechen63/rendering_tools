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
def compute_CSIM(args, path):
    # path = pathlib.Path(path)
    model = face_model.FaceModel(args)

    # files = list(path.glob('*real_image*.jpg')) + list(path.glob('*real_image*.png'))
    # files.sort()
    sims = []
    # # msssims = []
    path = pathlib.Path(path)
    fakefiles = list(path.glob('*fake*.mp4'))
    # fakefiles = list(path.glob(os.path.join('*','fake.mp4')))
    # fakefiles = list(path.glob(os.path.join('*', '*', 'test.mp4')))
    fakefiles.sort()
    t_files = []
    f_files = []
    for v in fakefiles[:5]:
        for f in read_videos(v, True, 2):
            f_files.append(f)
        for f in read_videos(v, True, 1):
            t_files.append(f)

    for i,d in enumerate(tqdm(f_files)):
            # if i == 10:
            #     break
        try:
                # real_path = str(d)
                # fake_path  = real_path.replace("_real_","_synthesized_")
            f_i = d
            # cv2.imwrite('/home/cxu-serve/p1/common/WSOL/tmp/false_{}.jpg'.format(i), f_i)
            f_i = cv2.resize(f_i,(112,112))
            f_i = model.get_input(f_i)
            f1 = model.get_feature(f_i)
            
            r_i = t_files[i]
            # cv2.imwrite('/home/cxu-serve/p1/common/WSOL/tmp/true_{}.jpg'.format(i), r_i)
            r_i = cv2.resize(r_i,(112,112))
            r_i = model.get_input(r_i)
            f2 = model.get_feature(r_i)
            dist = np.sum(np.square(f1-f2))
            sim = np.dot(f1, f2.T)
            # print(sim)
            # print (fake_path)
            # print(sim)

            sims.append(sim)
        except:
            sims.append(0.1)
        if i == 500:
            break

    print (sum(sims) / len(sims))

        

path = '/home/cxu-serve/p1/common/ablation/no_atten2' 
# path = '/home/cxu-serve/p1/common/experiment/nips_lrs/lrs_32_shot'
# path = args.path
compute_CSIM(args, path)

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