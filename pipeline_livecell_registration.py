#########################################################################################################
#########################################################################################################
import os
import sys
from numpy import *
from PIL import ImageFilter
from PIL import Image
import scipy
from scipy import optimize
from tifffile import *
from skimage import feature
from skimage import filters
from skimage import transform
import matplotlib.pyplot as plt
#########################################################################################################
#########################################################################################################
pathIn = "/DATA/lenstra_lab/in.brouwer/livecell_data/20200528/YTL1281B2-1-2-1_1hr_DMSO_galinduction_2/"
pathOut = "/DATA/lenstra_lab/in.brouwer/livecell_data/20200528/reg_YTL1281B2-1-2-1_1hr_DMSO_galinduction_2/"

if pathIn[-1] != "/": pathIn += "/"
if pathOut[-1] != "/": pathOut += "/"

zSlices = 9
frames = 240

pathIn = pathIn+"Pos0/"
allFiles = os.listdir(pathIn)
if "metadata.txt" in allFiles: allFiles.remove("metadata.txt")

if not os.path.exists(pathOut):
    os.makedirs(pathOut)

if not os.path.exists(pathOut+"Pos0/"):
    os.makedirs(pathOut+"Pos0/")

if not os.path.exists(pathOut+"Pos0/metadata.txt"):
    metadataold = pathIn+"metadata.txt"
    metadatanew =pathOut+"Pos0/metadata.txt"
    os.system("cp "+metadataold+" "+metadatanew)

#########################################################################################################
#########################################################################################################
# this part reads in the data (as a stack)
imMaxStack = []
for fn in range(0, frames):
    stack = [Image.open(pathIn+'img_%09d_GFP_GFP-mSc-filter_%03d.tif'%(fn,z)).convert('I') for z in r_[:zSlices]]
    imtmpR = (asarray([array(a.convert("I").getdata()).reshape(a.size) for a in stack]))*1.
    imMaxR = imtmpR.max(0)
    imMaxStack.append(imMaxR)
imMaxStack = array(imMaxStack)

#this part calculates the shifts wrt the first frame, after a gaussian blur of 5 pixels
regMaxStack = []
shifts = []
frame0 = filters.gaussian(imMaxStack[0],5)
for fn in range(1,frames):
    frame = filters.gaussian(imMaxStack[fn],5)
    shift =feature.register_translation(frame0,frame)[0]
    shifts.append(shift)

# this part calculates the registered max projection
transimage = []
transimage.append(imMaxStack[0])
for fn in range(1,frames):
    trans = [-shifts[fn-1][1],-shifts[fn-1][0]]
    image = imMaxStack[fn]
    transobject = transform.AffineTransform(translation = trans)
    transframe = transform.warp(image,transobject, mode = 'edge')
    transimage.append(transframe)

imsave(pathOut+"reg_max.tif", array(transimage).astype('uint16'), imagej=True)

#this part applies the shifts calculated above to the entire stack and writes the new datafiles
for fn in range(0,frames):
    stack = [Image.open(pathIn+'img_%09d_GFP_GFP-mSc-filter_%03d.tif'%(fn,z)).convert('I') for z in r_[:zSlices]]
    imtmpR = (asarray([array(a.convert("I").getdata()).reshape(a.size) for a in stack]))*1.
    if fn != 0:
        trans = [-shifts[fn-1][1],-shifts[fn-1][0]]
        for z in range(0,9):
            image = imtmpR[z]
            transobject = transform.AffineTransform(translation = trans)
            transframe = transform.warp(image, transobject, mode = 'edge')
            imsave(pathOut+'Pos0/img_%09d_GFP_GFP-mSc-filter_%03d.tif'%(fn,z), array(transframe).astype('uint16'), imagej=True)
    else:
        for z in range(0,9):
            image = imtmpR[z]
            imsave(pathOut+'Pos0/img_%09d_GFP_GFP-mSc-filter_%03d.tif'%(fn,z), array(image).astype('uint16'), imagej=True)
