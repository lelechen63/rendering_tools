cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools
conda activate face_alignment
python df2net_landmark_detect.py --conf ../chen_facestar.json
python df2net_landmark_detect.py --conf ../gir1_facestar.json
python df2net_landmark_detect.py --conf ../gir1_iphone.json
python df2net_landmark_detect.py --conf ../Israel_iphone.json
python df2net_landmark_detect.py --conf ../jason_facestar.json
python df2net_landmark_detect.py --conf ../kevyn_iphone.json
python df2net_landmark_detect.py --conf ../shugao_facestar.json
python df2net_landmark_detect.py --conf ../steve_iphone.json