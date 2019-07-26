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
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import colorsys
import glob
import shutil
import pdb

from scipy.optimize import minimize
from scipy.optimize import root

DATA_PATH = "./"
fx_rgb = 909.718
fy_rgb = 910.090
cx_rgb = 628.84
cy_rgb = 385.37

def drawResult(idx,res):
    img = cv2.imread("./color/color_%04d"%(idx) + ".jpg")
    dat = np.loadtxt("./matching/result_%04d"%(idx) + ".csv")
    if(len(dat.shape)< 2): return
    point_d = dat[:,:3]
    point_d[:,1] *= -1
    point_d[:,2] *= -1
    P = np.array([fx_rgb, 0, cx_rgb, 0, fy_rgb, cy_rgb, 0, 0, 1]).reshape((3,3))
    t = np.array(res.x)
    point_transformed = point_d + t
    _point = point_transformed
    _point[:,0] = _point[:,0]/_point[:,2]
    _point[:,1] = _point[:,1]/_point[:,2]
    _point[:,2] = _point[:,2]/_point[:,2]
    _point= _point.dot(P.T)
    point_conv = np.zeros([_point.shape[0],2])
    point_conv[:,0] = _point[:,0]
    point_conv[:,1] = _point[:,1]

    for i in np.arange(point_conv.shape[0]):
        cv2.circle(img,(int(point_conv[i,0]),int(point_conv[i,1])),2,(0,0,255),3)
    cv2.imwrite("./calib_result/result_%04d.jpg"%(idx),img)
    cv2.waitKey(10)

def getCr(center_x, center_y, radius):
    Cr = np.array([
        [1,0,-center_x],
        [0,1,-center_y],
        [-center_x, -center_y, center_x*center_x + center_y*center_y - radius*radius]
    ])
    return Cr

def getCd(center_x, center_y, center_z, radius):
    Cd = np.array([
        [1,0,0,-center_x],
        [0,1,0,-center_y],
        [0,0,1,-center_z],
        [-center_x, -center_y,-center_z, center_x*center_x + center_y*center_y+ center_z*center_z - radius*radius]
    ])
    return Cd

def func(param, dat):
    point_d = dat[:,:3]
    point_rgb = dat[:,4:6]
    radius_d = dat[:,3]
    radius_rgb = dat[:,6]
    scale = dat[:,7]
    point_d_mean = dat[:,8:11]
    #R_vec = np.array(param[:9])
    R_vec = np.array([1,0,0,0,1,0,0,0,1])
    R = R_vec.reshape((3,3))
    #t = np.array(param[9:12])
    t = np.array(param[:3])
    #scale_learn = param[3]
    P = np.array([fx_rgb, 0, cx_rgb, 0, fy_rgb, cy_rgb, 0, 0, 1]).reshape((3,3))

    #point_d[:,0] = 1/scale * point_d[:,0]
    #point_d[:,1] = 1/scale * point_d[:,1]
    #point_d[:,2] = 1/scale * point_d[:,2]
    #point_d[:,0] = scale_learn*(point_d[:,0]-point_d_mean[:,0]) + point_d_mean[:,0]
    #point_d[:,1] = scale_learn*(point_d[:,1]-point_d_mean[:,1]) + point_d_mean[:,1]
    #point_d[:,2] = scale_learn*(point_d[:,2]-point_d_mean[:,2]) + point_d_mean[:,2]

    #point_d = 1/scale*point_d.dot(R.T) + t
    #point_d[:,0] = 1/scale*(point_d[:,0]-point_d_mean[:,0]) + point_d_mean[:,0]
    #point_d[:,1] = 1/scale*(point_d[:,1]-point_d_mean[:,1]) + point_d_mean[:,1]
    #point_d[:,2] = 1/scale*(point_d[:,2]-point_d_mean[:,2]) + point_d_mean[:,2]
    point_transformed = point_d.dot(R.T) + t
    _point = point_transformed
    _point[:,0] = _point[:,0]/_point[:,2]
    _point[:,1] = _point[:,1]/_point[:,2]
    _point[:,2] = _point[:,2]/_point[:,2]
    _point= _point.dot(P.T)
    point_conv = np.zeros([_point.shape[0],2])
    point_conv[:,0] = _point[:,0]
    point_conv[:,1] = _point[:,1]

    diff = point_rgb - point_conv
    #_residual_points = np.sqrt(np.sum(diff**2))/diff.shape[0]
    _residual_points = np.sqrt(np.sum(diff**2))/diff.shape[0]
    #_residual_spheres = 0
    
    K = np.c_[R, t]
    _residual_spheres = 0        
    point_center = point_d.dot(R.T) + t
    for i in np.arange(point_rgb.shape[0]):
        Cr = getCr(point_rgb[i,0],point_rgb[i,1],radius_rgb[i])
        Cd = getCr(point_conv[i,0],point_conv[i,1],radius_d[i]*np.sqrt(fx_rgb*fy_rgb)/point_transformed[i,2])
        #Cd = getCd(point_d[i,0],point_d[i,1],point_d[i,2],radius_d[i])
        #Cd2r = P.dot(K).dot(Cd.dot(K.T).dot(P.T))        
        #diff_sphere = Cr.flatten()-Cd2r.flatten()
        diff_sphere = Cr.flatten()-Cd.flatten()
        _residual_spheres += np.sqrt(np.sqrt(np.sum(diff_sphere**2))/point_rgb.shape[0]) #because quadric
    #print(_residual_spheres)

    #residual = _residual_points + _residual_spheres/1000
    residual = _residual_points
    print(residual)
    
    return residual

def callibration():
    coeff_files = glob.glob("./matching/result_*.csv")
    coeff_files = np.sort(coeff_files)  
    
    nonzero_first_idx = 0
    for i in np.arange(coeff_files.shape[0]):
        if(len(np.loadtxt(coeff_files[i]).shape)> 1):
            nonzero_first_idx = i
            break

    dat = np.loadtxt(coeff_files[nonzero_first_idx])
    for i in np.arange(coeff_files.shape[0]-1)+1:
        if(len(np.loadtxt(coeff_files[i]).shape)> 1):
            dat = np.r_[dat,np.loadtxt(coeff_files[i])]

    dat[:,1] *= -1
    dat[:,2] *= -1

    #x_init = [1,0,0,0,1,0,0,0,1,0,0,0]
    x_init = [0.0,0.1,0.0]
    arg = (dat,)
    #result = minimize(func, x0=x_init, args=arg)
    #res = minimize(func, x0=x_init,args=arg, method='SLSQP', options={'xtol': 1e-10, 'disp': True, 'maxiter':10000})
    res = minimize(func, x0=x_init,args=arg, method='COBYLA', options={'xtol': 1e-10, 'disp': True, 'maxiter':10000})


    return res

if __name__ == "__main__":
    res = callibration()

    for i in np.arange(5)+4:
        drawResult(i,res)
        
    



