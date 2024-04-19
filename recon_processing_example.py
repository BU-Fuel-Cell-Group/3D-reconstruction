#IMPORT PACKAGES
import numpy
import time
import os
from PIL import Image
from scipy import ndimage, misc
import scipy
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import math
import skimage
from skimage import morphology
import skimage.measure
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import random
import skan #CITE!
from skan import Skeleton
import csv
from csv import writer

#DEFINE FUNCTIONS
def load_image(directory,filename,idx):
    imloaded=Image.open(directory+filename+'{:04}'.format(idx)+'.tif')
    return(imloaded)

def save_image(img,directory,filename,idx):
    img.save(directory+filename+'filled'+'{:04}'.format(idx)+'.tif') 

def phase_frac(arr):
    sizeArr=numpy.shape(arr)
    return(sum(sum(sum(arr)))/(sizeArr[0]*sizeArr[1]*sizeArr[2]))

def perc(barray,label):
    barray = (barray==label).astype(int)
    regions=skimage.measure.label(barray, connectivity=3)
    reg_props=regionprops(regions)
    reg_areas=numpy.zeros(len(reg_props))
    for i in range(len(reg_props)):
        reg_areas[i]=reg_props[i].area
    perc_index=(numpy.where(reg_areas == numpy.amax(reg_areas)))[0]
    perc_area=reg_props[int(perc_index)].area
    perc_frac=perc_area/sum(sum(sum(barray)))
    perc_locs=reg_props[int(perc_index)].coords
    print("Done")
    return(perc_frac,perc_locs)

def TPB_per_vol(arr,phase_values,vx_sz,dilations):
    sizeArr=numpy.shape(arr)
    tpb_array=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)

    for i in phase_values:
        barray=(arr==i).astype(int)
        barray_exp=scipy.ndimage.binary_dilation(barray,iterations=dilations).astype(barray.dtype)
        tpb_array=numpy.add(barray_exp,tpb_array)

    tpb_array=(tpb_array==len(phase_values)).astype(int)
    
    #Calculate total TPBs
    tpb_array=skimage.morphology.skeletonize_3d(tpb_array)
    perc_array=tpb_array #save array for percolation calculation
    numpy.save("tpb_array.npy", tpb_array)
    branch_data = skan.csr.Skeleton(tpb_array, spacing=[vx_sz[0],vx_sz[1], vx_sz[2]]) #voxel sizes are given along each direction
    tpb_length=sum(branch_data.path_lengths())
    tpb=tpb_length/((sizeArr[0]*sizeArr[1]*sizeArr[2]*(vx_sz[0]*vx_sz[1]*vx_sz[2])))
    
    perc_coords=(perc(perc_array,1))[1]
    
    perc_vol=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    for x in range(len(perc_coords)):
        perc_vol[(perc_coords[x,:])[0],(perc_coords[x,:])[1],(perc_coords[x,:])[2]]=1
        
    perc_tpb_array=skimage.morphology.skeletonize_3d(perc_vol)
    perc_branch_data=skan.csr.Skeleton(perc_tpb_array,spacing=[vx_sz[0],vx_sz[1], vx_sz[2]])
    perc_tpb_length=sum(perc_branch_data.path_lengths())
    perc_tpb=perc_tpb_length/((sizeArr[0]*sizeArr[1]*sizeArr[2]*(vx_sz[0]*vx_sz[1]*vx_sz[2])))
    
    return(tpb_array,tpb,perc_tpb, perc_tpb_array)

# Tortuosity--uses perc function
def tortuosity(arr,phase_label,walkers,steps):
    sizeArr=numpy.shape(arr)
    
    ## find the tortuosity in the percolated phase
    perc_results=perc(arr,phase_label)
    perc_coords=perc_results[1]
    
    perc_vol=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    for x in range(len(perc_coords)):
        perc_vol[(perc_coords[x,:])[0],(perc_coords[x,:])[1],(perc_coords[x,:])[2]]=1
    
    rp=numpy.zeros(steps)
    rpx=numpy.zeros(steps)
    rpy=numpy.zeros(steps)
    rpz=numpy.zeros(steps)
    for n in range(walkers):
        walk_seed_index=random.randint(0,len(perc_coords))
        walk_seed=(perc_results[1])[walk_seed_index]
        pos=walk_seed
        for s in range(steps):
            xchg=random.randint(-1,1)
            ychg=random.randint(-1,1)
            zchg=random.randint(-1,1)
            postemp=numpy.array([pos[0]+xchg,pos[1]+ychg,pos[2]+zchg])
            if (postemp[0] in range(sizeArr[0])) and (postemp[1] in range(sizeArr[1])) and (postemp[2]in range(sizeArr[2])) and (perc_vol[postemp[0],postemp[1],postemp[2]]==1):
                pos=postemp
            else:
                pos=pos
            rp[s]+=((pos[0]-walk_seed[0])**2+(pos[1]-walk_seed[1])**2+(pos[2]-walk_seed[2])**2)
            rpx[s]+=abs(pos[0]-walk_seed[0])**2
            rpy[s]+=abs(pos[1]-walk_seed[1])**2
            rpz[s]+=abs(pos[2]-walk_seed[2])**2
    
    # get displacement as a function of steps for all space, x, y and z
    rp=rp/walkers
    rpx=rpx/walkers
    rpy=rpy/walkers
    rpz=rpz/walkers
    
    ## Free space tortuosity
    rpfs=numpy.zeros(steps)
    rpxfs=numpy.zeros(steps)
    rpyfs=numpy.zeros(steps)
    rpzfs=numpy.zeros(steps)
    for n in range(walkers):
        walk_seed=numpy.array([random.randint(0,sizeArr[0]),random.randint(0,sizeArr[1]),random.randint(0,sizeArr[2])])
        pos=walk_seed
        for s in range(steps):
            xchg=random.randint(-1,1)
            ychg=random.randint(-1,1)
            zchg=random.randint(-1,1)
            postemp=numpy.array([pos[0]+xchg,pos[1]+ychg,pos[2]+zchg])
            if (postemp[0] in range(sizeArr[0])) and (postemp[1] in range(sizeArr[1])) and (postemp[2]in range(sizeArr[2])):
                pos=postemp
            else:
                pos=pos
            rpfs[s]+=((pos[0]-walk_seed[0])**2+(pos[1]-walk_seed[1])**2+(pos[2]-walk_seed[2])**2)
            rpxfs[s]+=abs(pos[0]-walk_seed[0])**2
            rpyfs[s]+=abs(pos[1]-walk_seed[1])**2
            rpzfs[s]+=abs(pos[2]-walk_seed[2])**2
            
    rpfs=rpfs/walkers
    rpxfs=rpxfs/walkers
    rpyfs=rpyfs/walkers
    rpzfs=rpzfs/walkers
    
    step_array=numpy.arange(1,steps+1)
    
    crp=(numpy.polyfit(step_array,rp,1))[0]
    crpx=(numpy.polyfit(step_array,rpx,1))[0]
    crpy=(numpy.polyfit(step_array,rpy,1))[0]
    crpz=(numpy.polyfit(step_array,rpz,1))[0]
    crpfs=(numpy.polyfit(step_array,rpfs,1))[0]
    crpxfs=(numpy.polyfit(step_array,rpxfs,1))[0]
    crpyfs=(numpy.polyfit(step_array,rpyfs,1))[0]
    crpzfs=(numpy.polyfit(step_array,rpzfs,1))[0]
    
    V_frac=perc_results[0]
    
    tort=(1/V_frac)*(crpfs/crp)
    tortx=(1/V_frac)*(crpxfs/crpx)
    torty=(1/V_frac)*(crpyfs/crpy)
    tortz=(1/V_frac)*(crpzfs/crpz)
    tortvals=[numpy.sqrt(tortx),numpy.sqrt(torty),numpy.sqrt(tortz),numpy.sqrt(tort)]
    
    return(tortvals,rp,rpx,rpy,rpz,step_array)


# Directional sizes
def bisector_size(arr,val,scale):
    sizeArr=numpy.shape(arr)
    x_sz=[]
    y_sz=[]
    z_sz=[]
    x_int=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    y_int=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    z_int=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    
    #find x sizes
    ct=0
    for z in range(sizeArr[2]):
        for y in range(sizeArr[1]):
            ct=0
            for x in range(sizeArr[0]):
                if arr[x,y,z]==val:
                    ct+=1
                else:
                    ct=0
                x_int[x,y,z]=ct
    for z in range(sizeArr[2]):
        for y in range(sizeArr[1]):
            for x in range(sizeArr[0]-1):
                if x_int[x,y,z]>0 and x_int[x+1,y,z]==0:
                    x_sz.append(x_int[x,y,z])
            if x_int[-1,y,z]>0:
                x_sz.append(x_int[-1,y,z])
                    
    x_size=numpy.mean(x_sz)*scale[0]
    
    
    #find y sizes
    ct=0
    for z in range(sizeArr[2]):
        for x in range(sizeArr[0]):
            ct=0
            for y in range(sizeArr[1]):
                if arr[x,y,z]==val:
                    ct+=1
                else:
                    ct=0
                y_int[x,y,z]=ct
    for z in range(sizeArr[2]):
        for x in range(sizeArr[0]):
            for y in range(sizeArr[1]-1):
                if y_int[x,y,z]>0 and y_int[x,y+1,z]==0:
                    y_sz.append(y_int[x,y,z])
            if y_int[x,-1,z]>0:
                y_sz.append(y_int[x,-1,z])
                    
    y_size=numpy.mean(y_sz)*scale[1]
                
        
    #find z sizes
    ct=0
    for y in range(sizeArr[1]):
        for x in range(sizeArr[0]):
            ct=0
            for z in range(sizeArr[2]):
                if arr[x,y,z]==val:
                    ct+=1
                else:
                    ct=0
                z_int[x,y,z]=ct
    for y in range(sizeArr[1]):
        for x in range(sizeArr[0]):
            for z in range(sizeArr[2]-1):
                if z_int[x,y,z]>0 and z_int[x,y,z+1]==0:
                    z_sz.append(z_int[x,y,z])
            if z_int[x,y,-1]>0:
                z_sz.append(z_int[x,y,-1])
                    
    z_size=numpy.mean(z_sz)*scale[2]
    
    #cat_sz=(numpy.multiply(x_sz*scale[0])+numpy.multiply(y_sz*scale[1])+numpy.multiply(z_sz*scale[2]))
    #cat_size=numpy.mean(cat_sz)
                
    return(1.5*x_size,1.5*y_size,1.5*z_size)


def surf2vol(volume,scale):
    [verts,faces,normals,values]=skimage.measure.marching_cubes(volume,spacing=(scale,scale,scale)) #includes outer surface
    A_outer=sum(sum(volume[:,:,0]))+sum(sum(volume[:,:,numpy.shape(volume)[2]-1]))+sum(sum(volume[:,0,:]))+sum(sum(volume[:,numpy.shape(volume)[1]-1,:]))+sum(sum(volume[0,:,:]))+sum(sum(pores[numpy.shape(volume)[0]-1,:,:]))

    #A=skimage.measure.mesh_surface_area(verts,faces)
    A=skimage.measure.mesh_surface_area(verts,faces)-A_outer*scale**2

    V_tot=sum(sum(sum(volume)))*scale**3

    return(A/V_tot)

def horz_quant(arr, phase, step, res):
    arr_size=numpy.shape(arr)
    slices = numpy.arange(0,arr_size[1],step)
    slices_um = [x * res for x in slices][1:]
    count_arr=[]
    for i in slices[1:]:
        slice_size = numpy.shape(arr[:,i-step:i,:])
        count=sum(sum(sum((arr[:,i-step:i,:]==phase))))
        count_arr.append(count/(slice_size[0]*slice_size[1]*slice_size[2]))
    return[slices_um,count_arr]

#LOAD DATA----------------------------------------------------
path = '/path/to/data/'
Labeled=ndimage.median_filter(numpy.load(path+'reconstruction.npy'), size=10)
sample='sample-name'
numpy.save('reconstruction-mf.npy',Labeled)

#CREATE CSV---------------------------------------------------
with open((path+'recon_data_'+sample+'.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Type", "Value","Unit"])

#write-in function
def report(data_list):
    with open((path+'recon_data_'+sample+'.csv'),'a') as fd:
        writer_object=csv.writer(fd)
        writer_object.writerow(data_list)


#GET DATA-----------------------------------------------------

#PHASE FRACTIONS
PoreLabel=3
NiLabel=1
YSZLabel=2

pores=(Labeled==PoreLabel).astype(int)
nickel=(Labeled==NiLabel).astype(int)
ysz=(Labeled==YSZLabel).astype(int)

pore_frac = phase_frac(pores)
ni_frac = phase_frac(nickel)
ysz_frac = phase_frac(ysz)

report(['Pore fraction', pore_frac, ''])
report(['Ni fraction', ni_frac, ''])
report(['YSZ fraction', ysz_frac, ''])

#PERCOLATION FRACTIONS
perc_pores=perc(pores,1)
report(['Perc fraction pores',perc_pores[0],''])
del perc_pores

perc_nickel=perc(nickel,1)
report(['Perc fraction Ni',perc_nickel[0],''])
del perc_nickel

perc_ysz=perc(ysz,1)
report(['Perc fraction YSZ',perc_ysz[0],''])
del perc_ysz

#TOTAL TPBs
resolutionX=0.0075 #in um
resolutionY=0.0075
resolutionZ=0.01
tpb=TPB_per_vol(Labeled.astype(int), [1,2,3], [resolutionX,resolutionY,resolutionZ], 3)
total_tpbs=tpb[1]

report(['Total TPBs',total_tpbs,'um^-2'])

#PERCOLATED TPBs
percolated_tpbs=tpb[2]

#PERCOLATED TPB FRACTION
percolated_tpb_fraction=percolated_tpbs/total_tpbs

report(['Percolated TPB fraction',percolated_tpb_fraction,''])

del tpb
del total_tpbs
del percolated_tpbs

#PORE TORTUOSITY
tort_pores=((tortuosity(pores,1,1000,10000))[0])[3]

report(['Pore Tortuosity',tort_pores,''])

#NI TORTUOSITY
tort_ni=((tortuosity(nickel,1,1000,10000))[0])[3]
report(['Ni Tortuosity',tort_ni,''])

#YSZ TORTUOSITY
tort_ysz=((tortuosity(ysz,1,1000,10000))[0])[3]
report(['YSZ Tortuosity',tort_ysz,''])

#PORE SIZE X
pore_sizes=bisector_size(pores,1,[resolutionX,resolutionY,resolutionZ])

report(['Pore size X',pore_sizes[0],'um'])

#PORE SIZE Y
report(['Pore size Y',pore_sizes[1],'um'])

#PORE SIZE Z
report(['Pore size Z',pore_sizes[2],'um'])


#NI SIZE X
ni_sizes=bisector_size(nickel,1,[resolutionX,resolutionY,resolutionZ])
report(['Ni size X',ni_sizes[0],'um'])

#NI SIZE Y
report(['Ni size Y',ni_sizes[1],'um'])

#NI SIZE Z
report(['Ni size Z',ni_sizes[2],'um'])

#YSZ SIZE X
ysz_sizes=bisector_size(ysz,1,[resolutionX,resolutionY,resolutionZ])
report(['YSZ size X',ysz_sizes[0],'um'])

#YSZ SIZE Y
report(['YSZ size Y',ysz_sizes[1],'um'])

#YSZ SIZE Z
report(['YSZ size Z',ysz_sizes[2],'um'])

#NI MIGRATION
ni_mig=horz_quant(Labeled, NiLabel, 100, 10)
numpy.asarray(ni_mig).tofile(path+'ni_migration_'+sample+'.csv', sep = ',')