import numpy
import time
import os
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import cv2
import pandas as pd

#nzidx is are nonzero indices in the image (found with numpy.argwhere(a)), s is the distance away you want to check
def nearest_nonzero_idx(a,x,y,nzidx,s):
    #nzidx = nzidx[~(nzidx == [x,y]).all(1)] #this would check that the index fed is nonzero, but don't need to do this if we have already filtered for zero points
    nzidx_near = nzidx[((nzidx[:,0] > (x-s)) & (nzidx[:,0] < (x+s)) & (nzidx[:,1] > (y-s)) & (nzidx[:,1] < (y+s)))] #only look for nearest nonzero points close to point of interest
    return nzidx_near[((nzidx_near - [x,y])**2).sum(1).argmin()]

def load_image(directory, name, idx):
    imloaded=Image.open(directory+name+'{:04}'.format(idx)+'_seg.tiff')
    return(numpy.asarray(imloaded))

def save_image(arr,directory,filename,idx):
    img = Image.fromarray(arr)
    img.save(directory+filename+'filled'+'{:04}'.format(idx)+'.tiff') 
    
#labelSlicRE is the image to correct, fIndexRE is the label being falsely identified
def remove_edges(labelSliceRE,fIndexRE,kernelSizeRE,cyclesRE):
    timers=[]
    
    start=time.time()
    [sliceSize1RE, sliceSize2RE]=numpy.shape(labelSliceRE)
    outputImRE=labelSliceRE

    kernelRE = numpy.ones((kernelSizeRE, kernelSizeRE), numpy.uint8) #kernel is related to the amount of erosion/dilation per cycle
    
    fSliceRE = (labelSliceRE==fIndexRE).astype(float) #pull out only pixels that correspond to the phase falsely identified as boundaries
    stop=time.time()
    timers.append('Get boundary phase:'+str(stop-start))
    
    start=time.time()
    workingImRE=fSliceRE #
    
    workingImRE = cv2.erode(workingImRE, kernelRE, iterations=cyclesRE) 
    workingImRE = cv2.dilate(workingImRE, kernelRE, iterations=cyclesRE)
    stop=time.time()
    timers.append('Erode/dilate:'+str(stop-start))
    
    start=time.time()
    edgesRE=fSliceRE-workingImRE
    edgesRE_invert=1-edgesRE
    withzeros=numpy.multiply(outputImRE,edgesRE_invert)
    
    filledIm = numpy.zeros([sliceSize1RE,sliceSize2RE])

    [zeroXs,zeroYs] = numpy.where(withzeros == 0)
    stop=time.time()
    timers.append('Get boundaries:'+str(stop-start))
    
    start=time.time()
    nonzeros = numpy.argwhere(withzeros)
    for i in range(len(zeroXs)):
        nonzero_coord = nearest_nonzero_idx(withzeros,zeroXs[i],zeroYs[i],nonzeros,5)
        filledIm[zeroXs[i],zeroYs[i]]=withzeros[nonzero_coord[0],nonzero_coord[1]]
    
    out=numpy.add(filledIm,withzeros)
    stop=time.time()
    timers.append('Fill boundaries:'+str(stop-start))
    
    return(out)

im_directory = r'/path/to/image/directory'
save_directory = r'/path/to/save/directory/'
im_name='imagename_'

i = int(os.getenv('SGE_TASK_ID'))-1

imtofill=load_image(im_directory,im_name,i)
imfilled = remove_edges(imtofill,2,5,1)
save_image(imfilled,save_directory,im_name,i)
