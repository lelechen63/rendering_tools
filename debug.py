import numpy as np 
import cv2



def load_img(img_path,lm_path):

	image = cv2.imread(img_path)
	lm = np.loadtxt(lm_path)
    print (image.shape)
    print (lm.shape)
    for i in range(5):
        land = lm[i]
        cv2.circle( image, (int( land[ 0 ] + 0.5), int( land[ 1 ] + 0.5 )), 2, (255, 0, 0), -1 )
    cv2.imwrite('./gg.png', image)
	# return image,lm

load_img('/mnt/ssd0/project/lchen63/cvpr2021/Deep3DFaceReconstruction/input/000002.jpg','/mnt/ssd0/project/lchen63/cvpr2021/Deep3DFaceReconstruction/input/000002.txt' )
