import os
import cv2 
import face_alignment
import numpy as np
import argparse
import json
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def landamrk_extract():

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
    elif datatype == "iphone":
        cams = ['iPhone']

    img_folder = os.path.join(  base_dir, datatype, pid, vid  )
    
    output_path = os.path.join(  base_dir, datatype, pid, vid , 'df2net'  )
    if not os.path.exists(output_path):
        os.mkdir( output_path)

    output_path = os.path.join(  base_dir, datatype, pid, vid , 'df2net' ,'lands' )
    if not os.path.exists(output_path):
        os.mkdir( output_path)

    
    for cam in cams:
        img_dir = os.path.join( img_folder, cam)
        # first time:
        tmp = os.listdir(img_dir)
        img_list =[]
        for t in tmp:
            if t[-3:] =='npy':
                img_list.append(t)
        img_list.sort()
     
        out_dir = os.path.join( output_path, cam  )
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for img_p in img_list:
            print (os.path.join( img_dir  ,img_p  ))
            raw_im = np.load( os.path.join( img_dir  ,img_p  ))
            raw_im = cv2.flip( raw_im, 0 )
            preds = fa.get_landmarks(raw_im)[0]
            preds = preds.astype(int)
            w = max( preds[:,0] ) - min( preds[:,0] ) 
            h = max( preds[:,1] ) - min( preds[:,1] ) 
            x_r = max( 0, min( preds[:,0] ) -  0.3*w )
            y_r = max( 0, min( preds[:,1] ) -  0.3*h )
            w_r = max ( min( raw_im.shape[0], w* 1.7), 0  )
            h_r  =max ( min( raw_im.shape[1], h* 1.7), 0  )
            w_r = h_r =min( w_r, h_r)

            roi_color = raw_im[ int(y_r):int(h_r + y_r),int(x_r):int(x_r+w_r)]
            roi_color =cv2.cvtColor(roi_color, cv2.COLOR_RGB2BGR)  
            img = cv2.resize(roi_color,(512,512))
            cv2.imwrite( os.path.join(   out_dir , img_p[:-4] + '_crop.png' ), img )
            img = cv2.resize(img,(224,224))
            preds = fa.get_landmarks(img)
            lmark_save_path = os.path.join(   out_dir , img_p[:-4] + '.npy' )
            np.save( lmark_save_path, preds[0])

landamrk_extract()
