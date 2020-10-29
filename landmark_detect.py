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


    base_dir = '/u/lchen63/cvpr2021/cvpr2021/data/data'
    # load data and prepare dataset
    pid = 'girl1'
    vid = "2020-10-19-12-05-51_leftside1"
    datatype = 'facestar'
    if datatype == "facestar":
        cams = ['cam00', 'cam01']
    img_folder = os.path.join(  base_dir, datatype, pid, vid  )
    
    output_path = os.path.join(  base_dir, datatype, pid, vid , 'deep3dfaceR'  )
    if not os.path.exists(output_path):
        os.mkdir( output_path)

    output_path = os.path.join(  base_dir, datatype, pid, vid , 'deep3dfaceR' ,'lands' )
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
            img = cv2.flip( raw_im, 0 )
            # raw_im =cv2.cvtColor(raw_im, cv2.COLOR_RGB2BGR)   
            preds = fa.get_landmarks(img)
            landmark = preds[0]
            # we need 5 landmarks of the original input image including right eye, left eye, nose tip, right mouth corner and left mouth corner. 
            need_landmark = []   
            # right eye
            need_landmark.append( (landmark[42] + landmark[45]) /2   )
            # left eye
            need_landmark.append( (landmark[36] + landmark[39]) /2   )
            # nose tip
            need_landmark.append( landmark[33]  )
            #right mouth corner
            need_landmark.append( landmark[54]  )
            #left mouth corner
            need_landmark.append( landmark[48]  )


            need_landmark = np.asarray(need_landmark)

            print(need_landmark.shape)

            lm = need_landmark
            image = img

            for i in range(5):
                land = lm[i]
                
                cv2.circle( image, (int( land[ 0 ] + 0.5), int( land[ 1 ] + 0.5 )), 2, (255, 0, 0), -1 )
                image = cv2.putText(image, str(i), (int( land[ 0 ] + 0.5), int( land[ 1 ] + 0.5 )), 2,  
                        1, (0, 0, 255), 1, cv2.LINE_AA) 
            cv2.imwrite('./gg.png', image)

            

            lmark_save_path = os.path.join(   out_dir , img_p[:-4] + '.npy' )
            np.save( lmark_save_path, need_landmark)
           
landamrk_extract()
