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
    else:
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
            raw_im =cv2.cvtColor(raw_im, cv2.COLOR_RGB2BGR)
               
            raw_gray = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)

            dets = face_cascade.detectMultiScale(raw_gray, 1.3, 5)
            if not isinstance(dets,tuple):
                for (x,y,w,h) in dets:
                    x_r = int(np.max((0,min(raw_im.shape[0],x-w*0.2))))
                    y_r = int(np.max((0,min(raw_im.shape[1],y-h*0.2))))
                    w_r = int(np.max((0,min(raw_im.shape[0],w*1.5))))
                    h_r = int(np.max((0,min(raw_im.shape[0],h*1.5))))
                    roi_color = raw_im[ y_r:h_r+y_r,x_r:x_r+w_r]
                    img = cv2.resize(roi_color,(224,224))
                    cv2.imwrite( 'gg.png' , img)
                    preds = fa.get_landmarks(img)
                    lmark_save_path = os.path.join(   out_dir , img_p[:-4] + '.npy' )
                    np.save( lmark_save_path, preds[0])
            else:
                print ('guickdfjkdjfdjkfd!!!!!')
                print (img_p)
landamrk_extract()
