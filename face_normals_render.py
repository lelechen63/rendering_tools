import numpy as np
import imageio
import matplotlib.pyplot as plt
import argparse
import json
import os

res =  256
def render_single(image_path ,mask_path , normal_path, output_path, light_dir = [0, 0, 1], light_intensity = 0.6):
    mask = imageio.imread(mask_path)
    normal = imageio.imread(normal_path)

    img = imageio.imread(image_path) / 255.0
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    normal = normal[:,:,:3] # only use the rgb
    
    mask = mask / 255.0
    normal = normal / 255.0
    normal = 2.0 * normal - 1

    light_dir = np.array(light_dir)
    light_dir = light_dir / np.linalg.norm(light_dir)

    cosine = normal.dot(light_dir)
    shading = light_intensity * cosine + 0.3 * light_intensity * cosine**9.0 + 0.2
    shading = shading * mask 
    shading =  shading.reshape(res, res, 1)
    shading  = np.repeat(shading, 3, axis = 2)
    print (shading.max() , img.max())
    mask =  mask.reshape(res, res, 1)
    mask  = np.repeat(mask, 3, axis = 2)
    output = shading + img * (1-mask)
    imageio.imsave(output_path, output)



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
    else:
        cams = ['iPhone']

    output_path = os.path.join(  base_dir, datatype, pid, vid , 'face_normals' )

    for cam in cams:
        out_dir = os.path.join( output_path, cam )
        # first time:
        tmp = os.listdir(out_dir)
        print (out_dir)
        
        img_list =[]
        for t in tmp:
            if t[-9:] =='_crop.jpg':
                img_list.append(t)
        img_list.sort()
        print (img_list)
        for img_p in img_list:
            image_path = os.path.join( out_dir  , img_p[:-9] +  '_crop.jpg')
            normal_path = os.path.join( out_dir  , img_p[:-9] +  '_normal.png')
            
            mask_path = os.path.join( out_dir  , img_p[:-9] +  '_musk.jpg')
            
            
            save_path =  os.path.join( out_dir  , img_p[:-4] +  '_without_tex.png')

            render_single( image_path, mask_path , normal_path, save_path )


if __name__ == '__main__':
    render_all()
    # render_single("Archive-normalnet/000001_normal.png", "Archive-normalnet/000001_musk.jpg", "fk.png")
