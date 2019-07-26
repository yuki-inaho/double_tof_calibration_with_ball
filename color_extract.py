#!/usr/bin/env python3
import sys
import glob
import re
import os
import math

import numpy as np
import scipy as sp
from scipy.linalg import eigh
try :
    import cv2
except :
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
    import cv2

from open3d import *
import matplotlib
import pdb
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import colorsys

import glob
import shutil
DATA_PATH = "./"
mean_radius=80


def getColorMask(color_image):
    img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    hsv_color1 = np.asarray([0, 0, 190])   # white!
    hsv_color2 = np.asarray([255, 255, 255])   # yellow! note the order

    mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
    #mask = cv2.bitwise_not(mask)
    return mask

def result2ColorImage(res_mask):
    color_res = np.zeros([res_mask.shape[0],res_mask.shape[1],3])
    #color_res[np.where(res_mask==-1)]  = 0
    color_res[np.where(res_mask == -1)[0], np.where(res_mask == -1)[1],:] = 0
    n_label = np.max(res_mask)

    for l in np.arange(n_label):
        h = (l+1)/(n_label)
        s = 1.0
        v = 1.0
        r,g,b= colorsys.hsv_to_rgb(h,s,v)

        print("{0},{1},{2}".format(r,g,b))
        color_res[np.where(res_mask == (l+1))[0], np.where(res_mask == (l+1))[1],0] = np.floor(r*255)
        color_res[np.where(res_mask == (l+1))[0], np.where(res_mask == (l+1))[1],1] = np.floor(g*255)
        color_res[np.where(res_mask == (l+1))[0], np.where(res_mask == (l+1))[1],2] = np.floor(b*255)

    color_res = color_res.astype(np.uint8)
    return color_res

def _batchSphereInfo(idx):
    #pinhole_camera_intrinsic = PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)

    color_img_name = "color_%04d"%(idx) + ".jpg"    
    depth_img_name = "depth_%04d"%(idx) + ".png"    
    color_image = cv2.imread(DATA_PATH + "color/" + color_img_name)
    depth_image = cv2.imread(DATA_PATH + "depth/" + depth_img_name, cv2.IMREAD_ANYDEPTH)    

    mask = getColorMask(color_image)
    mask[np.where(color_image[:,:,0] < 10)] = 0
    #kernel = np.ones((3,3),np.uint8)
    #mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)

    _color_image = np.zeros(color_image.shape)
    _color_image[mask.nonzero()[0], mask.nonzero()[1], :] = color_image[mask.nonzero()[0], mask.nonzero()[1], :]
    color_image = _color_image.astype(np.uint8)
    _color_image = np.zeros(color_image.shape)
    #_color_image[220:860,640:1280,:] = color_image[220:860,640:1280,:]
    _color_image[220:860,:,:] = color_image[220:860,:,:]
    color_image = _color_image.astype(np.uint8)

    
    mask = cv2.Canny(color_image, 100, 200)
    res_image = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    _circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT, 1.5, 100,
                                param1=90,param2=70,minRadius=0,maxRadius=0)

    _circles = np.uint16(np.around(_circles))
    circles = []
    for i in _circles[0,:]:
        diff = np.abs(i[2] - mean_radius)
#        diff = 0
        if(diff < 20):
            circles.append(i)
    circles = np.array([circles])
    for i in circles[0,:]:
        cv2.circle(res_image,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(res_image,(i[0],i[1]),2,(0,0,255),3)

    '''
    while cv2.waitKey(10) & 0xFF != ord('q') :
        cv2.imshow("test", res_image)
        cv2.imshow("mask", mask)        
    '''

    cv2.destroyAllWindows()
    coeff_mat = circles[0]
    coeff_name = "coeff_%04d"%(idx) + ".csv"    
    res_img_name = "result_%04d"%(idx) + ".png"    
    np.savetxt(DATA_PATH + "/hough_coeff/" + coeff_name, coeff_mat)
    cv2.imwrite(DATA_PATH + "/hough_result/" + res_img_name, res_image)


def batchSphereInfo():
    color_img_files = glob.glob("./color/color_*.jpg")
    color_img_files = np.sort(color_img_files)  

    coeff_save_path = './hough_coeff'
    if not os.path.exists(coeff_save_path):
        os.mkdir(coeff_save_path)
    else :
        shutil.rmtree(coeff_save_path)
        os.mkdir(coeff_save_path)

    result_save_path = './hough_result'
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    else :
        shutil.rmtree(result_save_path)
        os.mkdir(result_save_path)


    for color_name in color_img_files:
        color_name = os.path.basename(color_name)
        pattern = '.*?(\d+).*'
        result = re.search(pattern, color_name)
        zfill_idx = result.group(1)
        _batchSphereInfo(int(zfill_idx))

    #_batchSphereInfo(2)

if __name__ == "__main__":
    batchSphereInfo()


