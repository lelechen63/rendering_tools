#!/bin/bash

source /u/lchen63/miniconda2/etc/profile.d/conda.sh
conda init bash
RED='\033[0;31m'
NC='\033[0m' # No Color

# DF2net
ALG_NAME='DF2net'
conda activate base
conda activate pytorch1.1python3
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../rendering_tools
python df2net_render_no_tex.py --conf ../dani_iphone.json
cd ../rendering_tools

# MGCNet
ALG_NAME='MGCNet'
conda activate base
conda activate mgcnet
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../MGCNet
python test_image.py --conf ../dani_iphone.json
cd ../rendering_tools

# Deep3DR
ALG_NAME='Deep3DR'
conda activate base
conda activate pytorch1.1python3
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../rendering_tools
python deep3dfaceR_render_no_tex.py --conf ../dani_iphone.json
cd ../rendering_tools

# PRNet
ALG_NAME='PRNet'
conda activate base
conda activate pytorch1.1python3
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../rendering_tools
python prnet_render.py --conf ../dani_iphone.json
cd ../rendering_tools

# Face_normals
ALG_NAME='Face_normals'
conda activate base
conda activate pytorch1.1python3
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../rendering_tools
python face_normals_render.py --conf ../dani_iphone.json
cd ../rendering_tools

# RingNet
ALG_NAME='RingNet'
conda activate base
conda activate ringnet
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../RingNet
python demo.py --conf ../dani_iphone.json
cd ../rendering_tools

# 3DDFAv2
ALG_NAME='3DDFAv2'
conda activate base
conda activate 3ddfav2
echo -e "${RED}${ALG_NAME}, using ${CONDA_DEFAULT_ENV}${NC}"
cd ../3DDFA_V2
python demo.py --conf ../dani_iphone.json
cd ../rendering_tools

# DONE
echo -e "${RED}ALL DONE${NC}"
