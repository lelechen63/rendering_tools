import matplotlib.pyplot as plt
import imageio

algs = [
        'df2net', 
        'mgcnet', 
        'deep3dfaceR', 
        'prnet', 
        'face_normals', 
        'ringnet', 
        '3ddfav2'
        ]

img_name_tmp = [
        '_no_tex.png',
        '_overlayGeo.jpg',
        '._no_tex.png',
        '_without_tex.png',
        '_crop_without_tex.png',
        '_without_tex.png',
        '_3d.jpg'
        ]
img_id = '001201'
for i in range(1, 8):
    plt.subplot(2,4,i)
    plt.imshow(imageio.imread("/u/lchen63/cvpr2021/cvpr2021/data/data/facestar/girl1/2020-10-19-12-05-51_leftside1/{}/cam00/{}{}".format(algs[i-1], img_id, img_name_tmp[i-1])))
    plt.title(algs[i-1])
plt.show()
