import numpy as np 
import os 

def get_txt():
    
    for root, dirs, files in os.walk('/u/lchen63/cvpr2021/cvpr2021/data/data'):
        for file in files:
            if file.endswith(".png"):
                
                img_path = os.path.join(root, file)
                if 'deep3dfaceR' in img_path:
                    if '_no_tex.png' not in img_path:
                        print (img_path)
get_txt()