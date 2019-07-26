import pandas as pd
import sys
try :
    import cv2
except :
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
    import cv2
import numpy as np
import re
import os
import glob
import shutil
import pdb
from open3d import *
import pycpd
import scipy as sp
from scipy.spatial.transform import Rotation as Rot
from pycpd import deformable_registration, rigid_registration

from scipy.optimize import minimize

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from checkerboard import detect_checkerboard
#import optuna

CORNER_NUM = 5

PCD_DIR_LEFT = "./pcd/left"
PCD_DIR_RIGHT  = "./pcd/right"

left_pcd_data = glob.glob(PCD_DIR_LEFT + "/*.pcd")
left_pcd_data = np.sort(left_pcd_data)
right_pcd_data = glob.glob(PCD_DIR_RIGHT + "/*.pcd")
right_pcd_data = np.sort(right_pcd_data)

extrinsic_param = np.loadtxt("extrinsic_param.csv")
_Rot_init =  extrinsic_param[:3,:3].T
r = Rot.from_dcm(_Rot_init)
quaternion_init= r.as_quat()
translation_init = extrinsic_param[:3,3]

point_left_mat = np.zeros([len(left_pcd_data)*CORNER_NUM, 3])
point_right_mat = np.zeros([len(left_pcd_data)*CORNER_NUM, 3])

for i in np.arange(len(left_pcd_data)):
    left_pcd = read_point_cloud(left_pcd_data[i])
    right_pcd = read_point_cloud(right_pcd_data[i])
    point_left_mat[i*CORNER_NUM:(i+1)*(CORNER_NUM),:] = np.asarray(left_pcd.points)
    point_right_mat[i*CORNER_NUM:(i+1)*(CORNER_NUM),:] = np.asarray(right_pcd.points)

def func(param, point_left, point_right):
#def func(param, point_left, point_right, quaternion):
    quaternion = param[:4]
    r = Rot.from_quat(quaternion)
    Rot_mat = r.as_dcm()
    t = param[4:7]
    #t = param[:3]
    point_right_convd = point_right.dot(Rot_mat) + t    
    diff = point_left - point_right_convd

    #residual = np.sqrt(np.sum(diff**2))/diff.shape[0]
    residual = np.sqrt(np.sum(diff**2,axis=1))
    residual = np.sum(residual[np.where(residual <= np.median(residual))])/np.sum(residual <= np.median(residual))
    #residual = np.sum(residual)/residual.shape[0]
    #residual = np.sum(np.log(residual))/residual.shape[0]
    #residual[np.where(residual<c)] =  (residual[np.where(residual<c)]/c)**2
    #residual[np.where(residual>=c)] =  1
    return residual

    """最小化する目的関数"""
# パラメータが取りうる範囲
x_init = np.r_[quaternion_init,translation_init]
#x_init = np.r_[translation_init]
#arg = (point_left_mat, point_right_mat,quaternion_init,)
arg = (point_left_mat, point_right_mat,)
res = minimize(func, x0=x_init, args=arg, method='COBYLA', options={'xtol': 1e-10, 'disp': True, 'maxiter':10000})
residual = res.fun

quaternion = res.x[:4]
quaternion = quaternion_init
r = Rot.from_quat(quaternion)
Rot_mat_res = r.as_dcm()
t_res = res.x[4:7]
#t_res = res.x[:3]

r = Rot.from_dcm(Rot_mat_res) 
degrees = r.as_euler('yxz',degrees=True) 
test_conv = np.c_[Rot_mat_res.T, t_res[:,np.newaxis]] 
test_conv = np.r_[test_conv, np.array([0,0,0,1])[np.newaxis,:]]
np.savetxt("extrinsic_param.csv",test_conv)


