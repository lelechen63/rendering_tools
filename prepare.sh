# prepare landamrk for df2net
# conda activate face_alignment


# cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools
# python df2net_landmark_detect.py --conf ../chen_facestar.json
# python df2net_landmark_detect.py --conf ../gir1_iphone.json
# python df2net_landmark_detect.py --conf ../Israel_iphone.json
# python df2net_landmark_detect.py --conf ../jason_facestar.json
# python df2net_landmark_detect.py --conf ../kevyn_iphone.json
# python df2net_landmark_detect.py --conf ../shugao_facestar.json
# python df2net_landmark_detect.py --conf ../steve_iphone.json


#generate mat in df2net
# conda activate df2net

# cd /u/lchen63/cvpr2021/cvpr2021/DF2Net
# python demo.py --conf ../chen_facestar.json
# python demo.py --conf ../gir1_iphone.json
# python demo.py --conf ../Israel_iphone.json
# python demo.py --conf ../jason_facestar.json
# python demo.py --conf ../kevyn_iphone.json
# python demo.py --conf ../shugao_facestar.json
# python demo.py --conf ../steve_iphone.json

# python pointcloud2rawmesh.py --conf ../chen_facestar.json
# python pointcloud2rawmesh.py --conf ../gir1_iphone.json
# python pointcloud2rawmesh.py --conf ../Israel_iphone.json
# python pointcloud2rawmesh.py --conf ../jason_facestar.json
# python pointcloud2rawmesh.py --conf ../kevyn_iphone.json
# python pointcloud2rawmesh.py --conf ../shugao_facestar.json
# python pointcloud2rawmesh.py --conf ../steve_iphone.json


cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools
python deep3dfaceR_landmark_detect.py --conf ../chen_facestar.json
python deep3dfaceR_landmark_detect.py --conf ../gir1_iphone.json
python deep3dfaceR_landmark_detect.py --conf ../Israel_iphone.json
python deep3dfaceR_landmark_detect.py --conf ../jason_facestar.json
python deep3dfaceR_landmark_detect.py --conf ../kevyn_iphone.json
python deep3dfaceR_landmark_detect.py --conf ../shugao_facestar.json
python deep3dfaceR_landmark_detect.py --conf ../steve_iphone.json
