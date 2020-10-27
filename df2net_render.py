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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b",
                        type=int,
                        default=0)
    parser.add_argument("--",
                        type=str,
                        default='/home/cxu-serve/p1/lchen63/voxceleb/oppo/')

    parser.add_argument("--front_img_path",
                        type=str,
                        default='')
    # parser.add_argument("--front_frame_id",
    #                     type=int,
    #                     default=1)


    return parser.parse_args()
config = parse_args()
root = config.root


def load_obj(obj_file):
    vertices = []

    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                x = float(ts[1])
                y = float(ts[2])
                z = float(ts[3])
                r = float(ts[4])
                g = float(ts[5])
                b = float(ts[6])
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fx = int(ts[1]) - 1
                fy = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))

def setup_renderer():    
    renderer = sr.SoftRenderer(camera_mode="look", viewing_scale=2/res, far=10000, perspective=False, image_size=res, camera_direction=[0,0,-1], camera_up=[0,1,0], light_intensity_ambient=1)
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



def render_single_img(front_lmark_path = None ,  key_id = None):
    overlay = True
    # load cropped input_img
    input_image_path = "/u/lchen63/cvpr2021/cvpr2021/DF2Net/test_img/image0000_crop.png"
    input_img = cv2.imread(input_image_path)
    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("/u/lchen63/cvpr2021/cvpr2021/DF2Net/out_obj/image0000.obj") # get unfrontalized vertices position
    # set up the renderer
    renderer = setup_renderer()
    
    fig = plt.figure()
    temp_path = '/u/lchen63/cvpr2021/cvpr2021/SoftRas/result/df2net'
    # if os.path.exists(temp_path):
    #     shutil.rmtree(temp_path)
    # os.mkdir(temp_path)
    face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")
    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    
       
    rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
    cv2.imwrite( temp_path +  "/gg.png", rgb_frame)  
    print (temp_path +  "/gg.png")
render_single_img()