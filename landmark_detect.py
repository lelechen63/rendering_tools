import os
import cv2 
import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


def landamrk_extract():

    base_dir = '/u/lchen63/cvpr2021/cvpr2021/data/data'
    # load data and prepare dataset
    pid = 'girl1'
    vid = "2020-10-19-12-05-51_leftside1"
    datatype = 'facestar'
    if datatype == "facestar":
        cams = ['cam00', 'cam01']
    img_folder = os.path.join(  base_dir, datatype, pid, vid  )
    
    output_path = os.path.join(  base_dir, datatype, pid, vid , 'lands' )
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
            img = np.load( img_dir  ,img_p  )

            img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            preds = fa.get_landmarks(img)
            print (type(preds))
            print (preds)
            return 0

landamrk_extract()
