import sys
import soft_renderer as sr
import imageio
from skimage.transform import warp
from skimage.transform import AffineTransform
import numpy as np
import cv2
from scipy.io import loadmat, savemat

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
import json

res = 224


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



def render_single_img(  obj_path, mat_path , save_path):
    # load images
    mat_dic = loadmat(mat_path)
    print (mat_dic.keys())
    recon_img = mat_dic['recon_img']


    cropped_img = mat_dic['cropped_img']
    cropped_img  =cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)  

    rgb_frame =  (recon_img).astype(int)[:,:,:-1][...,::-1]


    # # load the original 3D face mesh then transform it to align frontal face landmarks
    # vertices_org, triangles, colors = load_obj(obj_path) # get unfrontalized vertices position
    # # set up the renderer
    # renderer = setup_renderer()
        
    # # render with texture
    # face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")

    # image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, np.uint8
    # rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
    mask = rgb_frame[:,:,0].reshape(res,res,1)
    mask_n = mask.sum(2)
    mask_n[mask_n!=0]=1
    mask = mask_n.reshape(res,res, 1)
    mask = np.repeat(mask, 3, axis = 2)
    final_output = cropped_img * (1 - mask) + mask * rgb_frame
    save_img = cv2.append( cropped_img, final_output)
    cv2.imwrite(save_path, save_img)  

def render_all():
    parser = argparse.ArgumentParser(description='PyTorch Face Reconstruction')
    parser.add_argument( '--conf', type = str, default = '' )
    global args
    args = parser.parse_args()
    conf_path = args.conf
    if conf_path == '':
        print( 'Error: please specificy configure path:' )
        print( '--conf CONF_PATH' )
        exit()

    # Load config
    with open( conf_path, 'r' ) as json_data:
        config = json.load( json_data )
    base_dir = config['basedir']
    pid = config['pid']
    vid = config['vid']
    datatype = config['datatype']
    if datatype == "facestar":
        cams = ['cam00', 'cam01']

    output_path = os.path.join(  base_dir, datatype, pid, vid , 'deep3dfaceR' )

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
            obj_path = os.path.join( out_dir  , obj[:-9] +  '_mesh.obj')
            print (obj_path)
            mat_path = os.path.join( out_dir  , obj[:-9] +  '.mat')

            save_path =  os.path.join( out_dir  , obj[:-9] +  '.png')

            render_single_img(  obj_path, mat_path , save_path)

render_all()