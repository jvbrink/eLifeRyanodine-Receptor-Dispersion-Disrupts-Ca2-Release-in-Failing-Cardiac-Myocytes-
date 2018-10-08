# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:29:16 2015

@author: MVLSTECH
"""
from pylab import imshow,plot,show,waitforbuttonpress,clf,subplot,title,connect,close,subplots,draw
import numpy as np
import easygui
from matplotlib.widgets import LassoSelector, RectangleSelector
from PyQt4 import QtCore, QtGui
import cv2
import tifffile
from scipy import ndimage,spatial
from skimage import measure
from skimage.filters import threshold_otsu
import pandas as pd
from glob import glob
import otsumod

def gui_fname(dir=None):
    """Select a file via a dialog and returns the file name.
    """
    if dir is None: dir ='./'
    fname =QtGui.QFileDialog.getOpenFileName(None, "Select data file...", 
            dir, filter="All files (*);; SM Files (*.sm)")
    return fname
    
def get_mask(im):
    global v
    v = []    
    def onselect(verts):
        global v
        v = verts
    
    fig, ax = subplots()
    ax.imshow(im,vmax=3)
    mask= np.zeros(im.shape,np.uint8)
    
    lasso = LassoSelector(ax, onselect)
    while waitforbuttonpress()!=True:
        pass
    bbox=np.zeros((len(v),1,2),np.int64)
    for i in range(len(v)):
        if None in v[i]:
            bbox[i,0]=lastone
        else:
            bbox[i,0]=np.int64(v[i])
            lastone=np.int64(v[i])
    cv2.drawContours(mask,[bbox],0,1,-1)
    close()
    #lasso = None
    #ax.imshow(mask)
    #draw()
    #show()
    return mask
    
def findgroupedRyRs(RyRim,psize=30,dist=150.0):
    #construct distance map from RyR image (30nm pixel)   
    dist=np.float32(dist)    
    origshape=RyRim.shape
    #origRyRim=RyRim.copy()
    #scale to 10 nm pixels
    RyRim=cv2.resize(np.float32(RyRim),(np.float32(RyRim.shape[0]*(psize/10)),np.float32(RyRim.shape[1]*(psize/10))),interpolation=cv2.INTER_NEAREST)
    RyRim_dmap=cv2.distanceTransform(np.uint8(RyRim<1),cv2.cv.CV_DIST_L2,5)
    #RyRim_dmap=RyRim_dmap[0]
    tRyRim_dmap=np.uint8(RyRim_dmap<dist/20) #(70nm)
    lim,n=ndimage.label(tRyRim_dmap)
    grouped_RyRs=cv2.bitwise_and(np.float32(lim),np.float32(lim),mask=np.uint8(RyRim))
    return np.int16(np.rint(cv2.resize(np.float32(grouped_RyRs),origshape)))

def RyRs_per_cluster(lim,n):
    #find subclusters per group of RyRs
    # use kernel for object detection to allow diagonal linked regions
    s = np.ones((3,3),np.uint8)
    smalln_arr=np.zeros(n)
    perims_arr=np.zeros(n)
    slices=ndimage.find_objects(lim,n)
    for i in range(n):
        sclustlabeled_array,smalln_arr[i]=ndimage.label(lim[slices[i]]==i+1,s)
    return smalln_arr
    
def neighbor(im,pixsize=30):	
    s=np.ones((3,3),np.uint8)
    lim,n=ndimage.label(im>0,s)
    #fig = figure()
    #ax = fig.add_subplot(211)
    #ax.imshow(lim)
    props=measure.regionprops(lim)
    RyRpos=list()
    for j in range(n):
            RyRpos.append(props[j]['Centroid'])
    RyRpos=np.array(RyRpos)
    tree = spatial.cKDTree(RyRpos)
    neighbours=list()
    for j in range(n):
        neighbours.append(tree.query(np.array(RyRpos)[j],k=2)[0][1])
    neighbours=np.array(neighbours)
    return neighbours

def get_solidity_area(lim):
    props=measure.regionprops(lim)
    soliditylist=list()
    for j in range(lim.max()):
        soliditylist.append([props[j]['Area'],props[j]['Solidity'],props[j]['Eccentricity'],props[j]['major_axis_length']])
    return(np.array(soliditylist))

def get_props(lim):
    props=measure.regionprops(lim)
    soliditylist=list()
    for j in range(lim.max()):
        soliditylist.append([props[j]['area'],props[j]['solidity'],props[j]['eccentricity'],props[j]['convex_area'],props[j]['centroid'][0],props[j]['centroid'][1],props[j]['major_axis_length'],props[j]['minor_axis_length']])
    return(np.array(soliditylist))   
    
def getthreshmask(im,mask,cent=99.7):
    im=mask*im
    #threshold image
    #changed from 9 to 4
    centile= np.percentile(im[mask==1], cent)
    #centile=np.percentile(np.clip(im,0.1,centile),cent)
    #cut off pixels greater than 99.7%
    maskcut=mask*im<centile
    thresh = otsumod.masked_otsu(im,maskcut)
    return mask*im>thresh
    
# Find the region with the largest area
def getMaxArea(ryrim):
    region_list = measure.regionprops(ryrim)
    
    maxArea = None
    for property in region_list:       
        if maxArea is None:
            maxArea = property
        else:
            if property.area > maxArea.area:
                maxArea = property
    return maxArea

def getDict(ryrim):
    props = measure.regionprops(ryrim)
    proplist=[]
    for prop in props:
        if prop is None:
          Dict = {'area'               :  0}
        else:
          Dict = {'label'              :  prop.label,
                         
                         'centroid_row'       :  prop.centroid[0],          # 0D:  location
                         'centroid_col'       :  prop.centroid[1],                     
                         
                         'diameter_equivalent':  prop.equivalent_diameter,  # 1D
                         'length_minor_axis'  :  prop.minor_axis_length, 
                         'length_major_axis'  :  prop.major_axis_length,
                         'ratio_eccentricity' :  prop.eccentricity,
                         'perimeter'          :  prop.perimeter,
                         'orientation'        :  prop.orientation,          # ranges from -pi/2 to pi/2 
                         
                         'area'               :  prop.area,                 # 2D
                         'area_convex'        :  prop.convex_area,
                         'area_filled'        :  prop.filled_area,
                         'box_min_row'        :  prop.bbox[0],
                         'box_max_row'        :  prop.bbox[2],
                         'box_min_col'        :  prop.bbox[1],
                         'box_max_col'        :  prop.bbox[3],
                         'ratio_extent'       :  prop.extent,
                         'ratio_solidity'     :  prop.solidity,                  
                         
                         'inertia_tensor_eigenvalue1':  prop.inertia_tensor_eigvals[0], 
                         'inertia_tensor_eigenvalue2':  prop.inertia_tensor_eigvals[1],
                         
                         'moments_hu1'        :  prop.moments_hu[0],        # translation, scale and rotation invariant
                         'moments_hu2'        :  prop.moments_hu[1],
                         'moments_hu3'        :  prop.moments_hu[2],
                         'moments_hu4'        :  prop.moments_hu[3],
                         'moments_hu5'        :  prop.moments_hu[4],
                         'moments_hu6'        :  prop.moments_hu[5],
                         'moments_hu7'        :  prop.moments_hu[6],
                         
                         'euler_number'       :  prop.euler_number,         # miscellaneous
                         }
        proplist.append(Dict)
    
    
                               
        
    return proplist

if __name__ == "__main__":
    #fn=easygui.fileopenbox(default=u"C:/Users/MVLSTECH/Documents/Terje's 2nd attempt/")
    #filelist=glob(path.dirname(fn)+'/*mask.npy')
    #adir="C:/Users/MVLSTECH/Documents/Terje's 2nd attempt/"
    #filelist=glob(adir+'*/*/*/*mask.npy')
    #
    batch=True
    if batch==True:
        adir="C:/Users/MVLSTECH/Documents/Terje's 2nd attempt/"
        filelist=glob(adir+'*/*/*mask.npy')    
        filelist=filelist+glob(adir+'*/*mask.npy')    
        filelist=filelist+glob(adir+'*/*/*/*mask.npy')
    #
    else:
        fn=easygui.fileopenbox(default=u"C:/Users/MVLSTECH/Documents/Terje's 2nd attempt/")
    if batch==True:    
        columns2=['filename','ryr no','xpos','ypos','Cluster size','CRU size','Solidity','Eccentricity','N distance','Clusters per CRU']
        df2=pd.DataFrame(columns2)
        df3=pd.DataFrame(columns2)
        for i in range(len(filelist)):
            #removed for batch
            #fn=gui_fname(dir=u"C:/Users/MVLSTECH/Documents/Terje's RyR super-res")
            columns=['RyRdist thresh','nClusters','nCRUs','Cluster size','CRU size','Solidity','Eccentricity','N distance','Clusters per CRU', 'Major axis length']
            bigcolumns=['RyRdist thresh','nClusters','nCRUs','Cluster size','CRU size','Solidity','Eccentricity','N distance','Clusters per CRU', 'Major axis length']
            df = pd.DataFrame(columns=columns)
            bigdf = pd.DataFrame(columns=bigcolumns)
            #added for batch        
            fn=filelist[i].replace('mask.npy','.tif')
            im=tifffile.imread(fn)    
            #im=ndimage.gaussian_filter(im,1)*3
            #im=im*9 changed to 3
            im=im*9
            im=cv2.resize(im,(im.shape[0]//3,im.shape[1]//3))        
            #mask=get_mask(im)
            maskname=(fn.lower()).replace('.tif','mask')
            #added for batch
            mask=np.load(maskname+'.npy')
            im=mask*im
            #threshold image
            #changed from 9 to 4
            thresh = threshold_otsu(im)
            im=getthreshmask(im,mask)
            #im=im>5
            np.save(maskname,mask)
            #save thresholded images
            tifffile.imsave(u"C:/Users/MVLSTECH/Documents/Terje's 2nd attempt/"+str(i)+".tif",np.uint8(im*255))
            #for ryrdistthresh in np.arange(100,160,10):
            for ryrdistthresh in [100,150]:
                #threshold image to detect ryr
                ryrim=findgroupedRyRs(im,dist=ryrdistthresh)      
                a=ndimage.sum(ryrim>0,ryrim,np.arange(1,ryrim.max()+1))
                
                nn=neighbor(im,30)
                #make image excluding small ryr clusters
                bigryrim=ryrim.copy()
                #get cluster sizes
                sizes = ndimage.sum(bigryrim>0, bigryrim, range(bigryrim.max() + 1))
                #make mask of small sized ones 
                mask_size=sizes<5
                #find clusters to remove
                remove_pixel=mask_size[bigryrim]
                bigryrim[remove_pixel] = 0
                #get new labels and reassign
                labels=np.unique(bigryrim)
                bigryrim = np.searchsorted(labels, bigryrim)
                
                #
                s = np.ones((3,3),np.uint8)
                bignn=neighbor(bigryrim,30)
                lim,n=ndimage.label(im,s)
                bigsa=get_solidity_area(bigryrim)
                ungroupedryrs=ndimage.sum(np.float32(im),labels=lim,index=np.arange(lim.max())+1)
                print n
                print ryrim.max()
                print ungroupedryrs.mean()
                print np.mean(nn)
                RPC=RyRs_per_cluster(ryrim,ryrim.max())
                bigRPC=RyRs_per_cluster(bigryrim,bigryrim.max())
                print np.mean(RPC)
                sa=get_solidity_area(ryrim)
                print np.mean(sa[:,1])
                print np.mean(sa[:,0])
                #filenames for histograms 
                
                statsname=(fn.lower()).replace('.tif','stats')
                bigstatsname=(fn.lower()).replace('.tif','bigstats')
                #calculate and save histograms for all parameters
                cruryrhistname=(fn.lower()).replace('.tif','cruryrhist.csv')
                b,c=np.histogram(a,bins=a.max(),range=(0.5,a.max()+0.5))
                np.save(cruryrhistname,b)
                #
                ryrhistname=(fn.lower()).replace('.tif','ryrhist.csv')
                b,c=np.histogram(ungroupedryrs,bins=ungroupedryrs.max(),range=(0.5,ungroupedryrs.max()+0.5))
                np.save(ryrhistname,b)
                #
                solidityhistname=(fn.lower()).replace('.tif','solidityhist.csv')
                b,c=np.histogram(sa[:,1],bins=50,range=(-0.02,1.02))
                np.save(solidityhistname,b)
                #
                bigsolidityhistname=(fn.lower()).replace('.tif','bigsolidityhist.csv')
                b,c=np.histogram(bigsa[:,1],bins=50,range=(-0.02,1.02))
                np.save(bigsolidityhistname,b)
                #
                eccentricityhistname=(fn.lower()).replace('.tif','eccentricityhist.csv')
                b,c=np.histogram(sa[:,2],bins=50,range=(-0.02,1.02))
                np.save(eccentricityhistname,b)
                #
                bigeccentricityhistname=(fn.lower()).replace('.tif','bigeccentricityhist.csv')
                b,c=np.histogram(bigsa[:,2],bins=50,range=(-0.02,1.02))
                np.save(bigeccentricityhistname,b)
                #
                disthistname=(fn.lower()).replace('.tif','disthist.csv')
                b,c=np.histogram(nn,bins=50,range=(0.5,50.5))
                np.save(disthistname,b)
                #
                clusterspercruhistname=(fn.lower()).replace('.tif','clusterspercruhist.csv')
                b,c=np.histogram(RPC,bins=10,range=(0.5,10.5))
                np.save(clusterspercruhistname,b)
                #
                #['RyRdist thresh','nClusters','nCRUs','Cluster size','CRU size','Solidity','Eccentricity','N distance','Clusters per CRU']
                statsarray=np.array([ryrdistthresh,n,ryrim.max(),ungroupedryrs.mean(),sa[:,0].mean(),sa[:,1].mean(),sa[:,2].mean(),nn.mean(),RPC.mean(),sa[:,3].mean()])
                bigstatsarray=np.array([ryrdistthresh,n,bigryrim.max(),ungroupedryrs.mean(),bigsa[:,0].mean(),bigsa[:,1].mean(),bigsa[:,2].mean(),bignn.mean(),bigRPC.mean(),bigsa[:,3].mean()])
                np.save(statsname,statsarray)
                np.save(bigstatsname,bigstatsarray)
                statscsv=(fn.lower()).replace('.tif','stats.csv')
                bigstatscsv=(fn.lower()).replace('.tif','bigstats.csv')                               
                df.loc[len(df)+1,]=statsarray
                bigdf.loc[len(df)+1,]=bigstatsarray
            d=getDict(ryrim)
            bigstatscsv=(fn.lower()).replace('.tif','bigstats.csv')
            allexcel=(fn.lower()).replace('.tif','allprops.xls')
            dfall=pd.DataFrame(d)
            dfall.to_excel(allexcel)
            bigstatscsv=(fn.lower()).replace('.tif','bigprops.csv')
            df.to_csv(statscsv)
            bigdf.to_csv(bigstatscsv)
            #removed for batch
            #tifffile.imshow(ryrim)
            # subclusters per supercluster, 
            #RyRs per supercluster 
            #Supercluster fragmentation. 
            #Nearest neighbour distances
            #Also, a nice histogram of single and supercluster sizes.
    else:
        columns=['RyRdist thresh','nClusters','nCRUs','Cluster size','CRU size','Solidity','Eccentricity','N distance','Clusters per CRU']
        df = pd.DataFrame(columns=columns)
        
        im=tifffile.imread(fn)    
        #im=ndimage.gaussian_filter(im,1)*3
        #im=im*9 changed to 3
        im=im*9
        im=cv2.resize(im,(im.shape[0]//3,im.shape[1]//3))        
        #mask=get_mask(im)
        maskname=(fn.lower()).replace('.tif','mask')
        #added for batch
        mask=np.load(maskname+'.npy')
        im=mask*im
        #threshold image
        #changed from 9 to 4
        thresh = threshold_otsu(im)
        im=im>thresh
        np.save(maskname,mask)
        #for ryrdistthresh in np.arange(100,160,10):
        for ryrdistthresh in [100,150]:
            #threshold image to detect ryr
            ryrim=findgroupedRyRs(im,dist=ryrdistthresh)      
            a=ndimage.sum(ryrim>0,ryrim,np.arange(1,ryrim.max()))
            b,c=np.histogram(a,bins=a.max(),range=(0.5,a.max()+0.5))
            histname=(fn.lower()).replace('.tif','hist')
            statsname=(fn.lower()).replace('.tif','stats')
            np.save(histname,b)
            nn=neighbor(im,30)
            s = np.ones((3,3),np.uint8)      
            lim,n=ndimage.label(im,s)
            ungroupedryrs=ndimage.sum(np.float32(im),labels=lim,index=np.arange(lim.max())+1)
            print n
            print ryrim.max()
            print ungroupedryrs.mean()
            print np.mean(nn)
            RPC=RyRs_per_cluster(ryrim,ryrim.max())
            print np.mean(RPC)
            sa=get_solidity_area(ryrim)
            print np.mean(sa[:,1])
            print np.mean(sa[:,0])
            statsarray=np.array([ryrdistthresh,n,ryrim.max(),ungroupedryrs.mean(),sa[:,0].mean(),sa[:,1].mean(),sa[:,2].mean(),nn.mean(),RPC.mean()])
            np.save(statsname,statsarray)
            statscsv=(fn.lower()).replace('.tif','stats.csv')            
            df.loc[len(df)+1,]=statsarray
        df.to_csv(statscsv)
        tifffile.imshow(ryrim)
        # subclusters per supercluster, 
        #RyRs per supercluster 
        #Supercluster fragmentation. 
        #Nearest neighbour distances
        #Also, a nice histogram of single and supercluster sizes.
