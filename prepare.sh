# prepare landamrk for df2net
# conda activate face_alignment


# cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools
# python df2net_landmark_detect.py --conf ../chen_facestar.json
# python df2net_landmark_detect.py --conf ../dani_iphone.json
# python df2net_landmark_detect.py --conf ../gir1_iphone.json
# python df2net_landmark_detect.py --conf ../Israel_iphone.json
# python df2net_landmark_detect.py --conf ../jason_facestar.json
# python df2net_landmark_detect.py --conf ../kevyn_iphone.json
# python df2net_landmark_detect.py --conf ../shugao_facestar.json
# python df2net_landmark_detect.py --conf ../steve_iphone.json


#generate mat in df2net
# conda activate df2net

# cd /u/lchen63/cvpr2021/cvpr2021/DF2Net

# python demo.py --conf ../dani_iphone.json
# python demo.py --conf ../chen_facestar.json
# python demo.py --conf ../gir1_iphone.json
# python demo.py --conf ../Israel_iphone.json
# python demo.py --conf ../jason_facestar.json
# python demo.py --conf ../kevyn_iphone.json
# python demo.py --conf ../shugao_facestar.json
# python demo.py --conf ../steve_iphone.json

# python pointcloud2rawmesh.py --conf ../dani_iphone.json
# python pointcloud2rawmesh.py --conf ../chen_facestar.json
# python pointcloud2rawmesh.py --conf ../gir1_iphone.json
# python pointcloud2rawmesh.py --conf ../Israel_iphone.json
# python pointcloud2rawmesh.py --conf ../jason_facestar.json
# python pointcloud2rawmesh.py --conf ../kevyn_iphone.json
# python pointcloud2rawmesh.py --conf ../shugao_facestar.json
# python pointcloud2rawmesh.py --conf ../steve_iphone.json


# cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools
# python deep3dfaceR_landmark_detect.py --conf ../dani_iphone.json
# python deep3dfaceR_landmark_detect.py --conf ../chen_facestar.json
# python deep3dfaceR_landmark_detect.py --conf ../gir1_iphone.json
# python deep3dfaceR_landmark_detect.py --conf ../Israel_iphone.json
# python deep3dfaceR_landmark_detect.py --conf ../jason_facestar.json
# python deep3dfaceR_landmark_detect.py --conf ../kevyn_iphone.json
# python deep3dfaceR_landmark_detect.py --conf ../shugao_facestar.json
# python deep3dfaceR_landmark_detect.py --conf ../steve_iphone.json


# deep 3DR
#conda activate mgcnet
# cd /u/lchen63/cvpr2021/cvpr2021/Deep3DFaceReconstruction
# python demo.py --conf ../dani_iphone.json
# python demo.py --conf ../chen_facestar.json
# python demo.py --conf ../gir1_iphone.json
# python demo.py --conf ../Israel_iphone.json
# python demo.py --conf ../jason_facestar.json
# python demo.py --conf ../kevyn_iphone.json
# python demo.py --conf ../shugao_facestar.json
# python demo.py --conf ../steve_iphone.json



# # 
# cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools

# python face_normals_landamrk_detect.py --conf ../dani_iphone.json
# python face_normals_landamrk_detect.py --conf ../chen_facestar.json
# python face_normals_landamrk_detect.py --conf ../gir1_iphone.json
# python face_normals_landamrk_detect.py --conf ../Israel_iphone.json
# python face_normals_landamrk_detect.py --conf ../jason_facestar.json
# python face_normals_landamrk_detect.py --conf ../kevyn_iphone.json
# python face_normals_landamrk_detect.py --conf ../shugao_facestar.json
# python face_normals_landamrk_detect.py --conf ../steve_iphone.json

#conda activate face_normals
# cd /u/lchen63/cvpr2021/cvpr2021/face_normals


# python tester.py --conf ../dani_iphone.json
# python tester.py --conf ../chen_facestar.json
# python tester.py --conf ../gir1_iphone.json
# python tester.py --conf ../Israel_iphone.json
# python tester.py --conf ../jason_facestar.json
# python tester.py --conf ../kevyn_iphone.json
# python tester.py --conf ../shugao_facestar.json
# python tester.py --conf ../steve_iphone.json

# conda activate prnet
# cd /u/lchen63/cvpr2021/cvpr2021/PRNet

# python demo.py --conf ../dani_iphone.json
# python demo.py --conf ../chen_facestar.json
# python demo.py --conf ../gir1_iphone.json
# python demo.py --conf ../Israel_iphone.json
# python demo.py --conf ../jason_facestar.json
# python demo.py --conf ../kevyn_iphone.json
# python demo.py --conf ../shugao_facestar.json
# python demo.py --conf ../steve_iphone.json


# cd /u/lchen63/cvpr2021/cvpr2021/rendering_tools
# python ours_render.py --conf ../dani_iphone.json
# python ours_render.py --conf ../chen_facestar.json
# python ours_render.py --conf ../gir1_iphone.json
# python ours_render.py --conf ../Israel_iphone.json
# python ours_render.py --conf ../jason_facestar.json
# python ours_render.py --conf ../kevyn_iphone.json
# python ours_render.py --conf ../shugao_facestar.json
# python ours_render.py --conf ../steve_iphone.json