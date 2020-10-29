import sys
import soft_renderer as sr
import imageio
from skimage.transform import warp
from skimage.transform import AffineTransform
import numpy as np
import cv2

import torch
import mmcv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import time
import os
import pickle
import shutil
import argparse
res = 512
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--b",
#                         type=int,
#                         default=0)
#     parser.add_argument("--root",
#                         type=str,
#                         default='/home/cxu-serve/p1/lchen63/voxceleb/oppo/')

#     parser.add_argument("--front_img_path",
#                         type=str,
#                         default='')


#     return parser.parse_args()
# config = parse_args()


def load_obj(obj_file):
    vertices = []

    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                y = float(ts[1])
                x = float(ts[2])
                z = float(ts[3])
                r = float(ts[4])
                g = float(ts[5])
                b = float(ts[6])
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fy = int(ts[1]) - 1
                fx = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))



def load_obj_without_color(obj_file):
    vertices = []

    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                y = float(ts[1])
                x = float(ts[2])
                z = float(ts[3])
                r = float(1)
                g = float(1)
                b = float(1)
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fy = int(ts[1]) - 1
                fx = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))

def setup_renderer():    
    renderer = sr.SoftRenderer(
        camera_mode="look", 
        viewing_scale=2/res, 
        far=10000, 
        perspective=False, 
        image_size=res, 
        camera_direction=[0,0,-1], 
        camera_up=[0,1,0],
        light_intensity_ambient=0.2, 
        light_color_ambient=[1,1,1],
        light_intensity_directionals=0.6, 
        light_color_directionals=[1,1,1],
        light_directions=[-1,0,-0.8]
    )
    renderer.transform.set_eyes([res/2, res/2, 6000])
    return renderer

def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [1,2])
    image = image.detach().cpu().numpy().transpose((1,2,0))
    image = np.clip(image, 0, 1)
    image = (255*image).astype(np.uint8)
    return image



def render_single_img( image_path, mask_path , obj_path, save_path):
    # overlay = True
    # load cropped input_img
    input_image_path = "/u/lchen63/cvpr2021/cvpr2021/DF2Net/test_img/image0000_ori.png"
    mask_n = cv2.imread("/u/lchen63/cvpr2021/cvpr2021/DF2Net/test_img/image0000_mask.png")
    input_img = cv2.imread(input_image_path)
    # print (input_img.max())
    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("/u/lchen63/cvpr2021/cvpr2021/DF2Net/out_obj/image0000.obj") 

    # set up the renderer
    renderer = setup_renderer()
    
    # fig = plt.figure()
    temp_path = './results/df2net'
    
    # render without texture
    # face_mesh = sr.Mesh(vertices_org, triangles, texture_type="vertex")

    # render with texture
    face_mesh = sr.Mesh(vertices_org, triangles)

    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]

    mask = rgb_frame[:,:,0]
    mask_n = mask_n.sum(2)
    mask_n[mask_n!=0]=1
    mask = mask_n.reshape(res,res, 1)
    mask = np.repeat(mask, 3, axis = 2)
    cv2.imwrite( temp_path +  "/mask.png", mask * 255)  
    final_output = input_img * (1 - mask) + mask * rgb_frame
    cv2.imwrite( temp_path +  "/conbined.png", final_output)  

def render_all():
    base_dir = '/u/lchen63/cvpr2021/cvpr2021/data/data'
    # load data and prepare dataset
    pid = 'girl1'
    vid = "2020-10-19-12-05-51_leftside1"
    datatype = 'facestar'
    if datatype == "facestar":
        cams = ['cam00', 'cam01']

    output_path = os.path.join(  base_dir, datatype, pid, vid , 'df2net' )

    for cam in cams:
        out_dir = os.path.join( output_path, cam )
        # first time:
        tmp = os.listdir(out_dir)
        print (out_dir)
        
        obj_list =[]
        for t in tmp:
            if t[-3:] =='obj':
                obj_list.append(t)
        obj_list.sort()
        print (obj_list)
        for obj in obj_list:
            obj_path = os.path.join( out_dir  , obj[:-4] +  '.obj')
            print (obj_path)



render_all()