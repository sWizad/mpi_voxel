import os
import glob
import sys

dataset = 'palm'
layers = 12
sublayers = 12
codename = 'model_muti_res.py'

invz = "-invz"
offset = "64"
if dataset in ['willow','path']:
    ref_img, scale = "0013", "0.29025"
elif dataset in ['titus','splash','glass']:
    ref_img, scale = "0013", "0.5"
elif dataset in ['airplants']:
    ref_img, scale = "2081", "0.127"
    invz, offset = "", "128"
elif dataset in ['pond']:
    ref_img, scale = "3056", "0.127"
    ref_img, scale = "3056", "0.1588" #640*480
    #offset = "128"
elif dataset in ['santarex']:
    ref_img, scale = "54618", "0.127"
    invz = ""
elif dataset in ['palm']:
    ref_img, scale = "038", "0.127" #512x384
    ref_img, scale = "038", "0.1588" #640*480



if(1):
    command = "python "+codename+" -dataset="+dataset+" -scale="+scale+" -ref_img="+ref_img
    command +=" -layers=%d -sublayers=%d "%(layers,sublayers)+invz
    command +=" -offset="+offset +" -subscale=8"
    print(command)
    os.system(command)
    #os.system(command + " -restart")
    #os.system(command + " -predict")
