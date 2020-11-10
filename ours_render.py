import numpy as np
import imageio
import matplotlib.pyplot as plt
import argparse
import json
import os

def render_single(image_path , output_path, light_dir = [0, 0, 1], light_intensity = 0.6):
    result =    imageio.imread(image_path) 
    print (result.shape)
    high = result.shape[0]
    width = int(result.shape[1] /3)
    normal = result[:,width * 2:, :]
    img = result[:,:width,:]/255.0

    mask = normal.copy()
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    mask[mask >0] =255
    normal = normal[:,:,:3] # only use the rgb
    
    mask = mask / 255.0
    normal = normal / 255.0
    normal = 2.0 * normal - 1

    light_dir = np.array(light_dir)
    light_dir = light_dir / np.linalg.norm(light_dir)

    cosine = normal.dot(light_dir)
    shading = light_intensity * cosine + 0.3 * light_intensity * cosine**9.0 + 0.2
    shading = shading * mask 
    shading =  shading.reshape(high, width, 1)
    shading  = np.repeat(shading, 3, axis = 2)
    print (shading.max() , img.max())
    mask =  mask.reshape(high, width, 1)
    mask  = np.repeat(mask, 3, axis = 2)
    output = shading + img * (1-mask)
    print (output_path)
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

    img_path_path = os.path.join(  base_dir, datatype, pid, vid , 'ours' )

    output_path = os.path.join(  base_dir, datatype, pid, vid , 'ours_render' )
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    for cam in cams:
        out_dir = os.path.join( output_path, cam )
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # first time:
        tmp = os.listdir(os.path.join( img_path_path, cam ))
        img_list =[]
        for t in tmp:
                img_list.append(t)
        img_list.sort()
        print (img_list)
        for img_p in img_list:
            image_path = os.path.join( img_path_path, cam  , img_p)
            
            save_path =  os.path.join( out_dir  , img_p[:-4] +  '_without_tex.png')

            render_single( image_path, save_path )


if __name__ == '__main__':
    render_all()
    # render_single("Archive-normalnet/000001_normal.png", "Archive-normalnet/000001_musk.jpg", "fk.png")
