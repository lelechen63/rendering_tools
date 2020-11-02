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
import json
res = 224
from scipy.io import loadmat, savemat

import sys
import soft_renderer as sr
import imageio
from skimage.transform import warp
from skimage.transform import AffineTransform
import numpy as np
import cv2

from scipy.io import loadmat, savemat

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


def setup_renderer():    
    renderer = sr.SoftRenderer(
        camera_mode="look", 
        far=50.0,
        near=0.01, 
        perspective=False,
        image_size=res, 
        camera_direction=[0,0,-1], 
        camera_up=[0,1,0],
        light_intensity_ambient=0.2, 
        light_color_ambient=[1,1,1],
        light_intensity_directionals=0.6, 
        light_color_directionals=[1,1,1],
        light_directions=[0,0,1]
    )
    renderer.transform.set_eyes([0, 0, 10])
    return renderer

def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [1,2])
    image = image.detach().cpu().numpy().transpose((1,2,0))
    print (image.max() , image.min(), '====')
    image = np.clip(image, 0, 1)
    image = (255*image).astype(np.uint8)
    return image

def render_single_img( mat_path , obj_path, save_path):
    # overlay = True
    # load cropped input_img
    mat_dic = loadmat(mat_path)
    cropped_img = mat_dic['cropped_img']
    # cropped_img  =cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)  
    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj(obj_path) 

    # set up the renderer
    renderer = setup_renderer()
        
    # render without texture
    face_mesh = sr.Mesh(vertices_org, triangles, texture_type="vertex")

    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    rgb_frame =  (image_render).astype(int)[:,:,:-1]

    rgb_frame = rgb_frame.transpose(1,0,2)[::-1, :, :]
    
    # plt.imshow(rgb_frame)
    # plt.show()

    mask_n = rgb_frame.sum(2)
    mask_n[mask_n!=0]=1
    mask = mask_n.reshape(res,res, 1)
    mask = np.repeat(mask, 3, axis = 2)
    final_output = cropped_img * (1 - mask) + mask * rgb_frame

    cropped_img  =cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(save_path, final_output)


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

            save_path =  os.path.join( out_dir  , obj[:-9] +  '._no_tex.png')
            
            render_single_img( mat_path , obj_path, save_path)


render_all()