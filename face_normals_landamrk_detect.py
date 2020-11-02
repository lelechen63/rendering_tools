import os
import cv2 
import face_alignment
import numpy as np
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
import argparse
import json

def landamrk_extract():
    #============================================================================
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


    img_folder = os.path.join(  base_dir, datatype, pid, vid  )
    
    output_path = os.path.join(  base_dir, datatype, pid, vid , 'face_normals'  )
    if not os.path.exists(output_path):
        os.mkdir( output_path)

    # landmark_path = os.path.join(  base_dir, datatype, pid, vid , 'face_normals' ,'lands' )
    # if not os.path.exists(landmark_path):
    #     os.mkdir( landmark_path)

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
            img = cv2.flip( raw_im, 0 )
            # raw_im =cv2.cvtColor(raw_im, cv2.COLOR_RGB2BGR)   
            preds = fa.get_landmarks(img)
            landmark = preds[0]
            
            shape = np.round(landmark).astype(int)
            # draw mask 
            print (shape.shape)
            print(shape)
            msk = np.zeros(img.shape, dtype=np.uint8)
            cv2.fillPoly(msk, [cv2.convexHull(shape)], (1,1,1))
                
            # crop & resize
            umin = np.min(shape[:,0]) 
            umax = np.max(shape[:,0])
            vmin = np.min(shape[:,1]) 
            vmax = np.max(shape[:,1])
                    
            umean = np.mean((umin,umax))
            vmean = np.mean((vmin,vmax))
                            
            l = round( 1.2 * np.max((umax-umin,vmax-vmin)))
                
            if (l > np.min(img.shape[:2])):
                l = np.min(img.shape[:2])
                    
            us = round(np.max((0,umean-float(l)/2)))
            ue = us + l
                
            vs = round(np.max((0,vmean-float(l)/2))) 
            ve = vs + l
                        
            if (ue>img.shape[1]):
                ue = img.shape[1]
                us = img.shape[1]-l
                
            if (ve>img.shape[0]):
                ve = img.shape[0]
                vs = img.shape[0]-l
                            
            us = int(us)
            ue = int(ue)  
                
            vs = int(vs)
            ve = int(ve)    
                
            img = cv2.resize(img[vs:ve,us:ue],(256,256))
            msk = cv2.resize(msk[vs:ve,us:ue],(256,256),interpolation=cv2.INTER_NEAREST)
                
            # save images 

            cv2.imwrite( os.path.join(out_dir , img_p[:-4] + '_musk.jpg' ) , msk * 255)
            img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   
            cv2.imwrite( os.path.join(out_dir , img_p[:-4] + '_crop.jpg' ), img)
           
landamrk_extract()
