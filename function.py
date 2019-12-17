import numpy as np
import scipy as sci
from numpy.random import *
from PIL import Image
import os
import copy
import sys

def extract_patch(im,patch_hight,patch_width):
    size=im.shape
    patch_hight_num=int(size[0]/patch_hight)
    patch_width_num=int(size[1]/patch_width)
    print(patch_hight_num)
    k=0
    patches=np.zeros(shape=(patch_hight_num*patch_width_num , patch_hight , patch_width),dtype=float)
    print(patches.shape)
    for i in range(patch_hight_num):
        for j in range(patch_width_num):
            patches[k,:,:]=im[i*patch_hight:(i+1)*patch_hight,j*patch_width:(j+1)*patch_width]
            k+=1
    return patches
