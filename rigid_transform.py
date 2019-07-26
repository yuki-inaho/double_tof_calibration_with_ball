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

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time



PCD_DIR_LEFT = "./pcd/left"
PCD_DIR_RIGHT  = "./pcd/right"

def callback(iteration, error, X, Y):
    print('Iteration: {:d} Error: {:06.4f}'.format(iteration, error))


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.2,0.8)
    ax.set_zlim(0,1.2)

    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def RigidTransform(Y, X):
#    Y = points_right
#    X = points_left

    muY = np.mean(Y,axis=0)
    muX = np.mean(X,axis=0)
    X_centrized = X - muX
    Y_centrized = Y - muY

    #reg = rigid_registration(**{ 'X': X_centrized, 'Y':Y_centrized })             
    reg = rigid_registration(**{ 'X': X, 'Y':Y })             
    reg.register(callback)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)            
    reg.register(callback)
    plt.show()
    '''

    permutate_idx = np.argmax(reg.P,axis=1)
    #resid =  X[permutate_idx]  - (Y.dot(reg.R.T) + muX - muY.dot(reg.R.T) + reg.t)

    scale= reg.s
    rotation = Rot.from_dcm(reg.R)
    quat = rotation.as_quat()


    #trans = muX - muY.dot(reg.R.T) + reg.t
    trans = reg.t
    return scale, quat, trans, reg.err 

left_pcd_data = glob.glob(PCD_DIR_LEFT + "/*.pcd")
left_pcd_data = np.sort(left_pcd_data)
right_pcd_data = glob.glob(PCD_DIR_RIGHT + "/*.pcd")
right_pcd_data = np.sort(right_pcd_data)

quaternion_list = []
translation_list = []
scale_list = []
err_list = []

for i in np.arange(len(left_pcd_data)):
    left_pcd = read_point_cloud(left_pcd_data[i])
    right_pcd = read_point_cloud(right_pcd_data[i])
    points_left = np.asarray(left_pcd.points)
    points_right = np.asarray(right_pcd.points)

    scale, quat, trans, err= RigidTransform(points_right, points_left)
    
    quaternion_list.append(quat)
    translation_list.append(trans)
    scale_list.append(scale)
    err_list.append(err)

#quaternion_list = np.asarray(quaternion_list)[np.where(err_list<np.median(err_list))[0]]
#translation_list = np.asarray(translation_list)[np.where(err_list<np.median(err_list))[0]]
#scale_list = np.asarray(scale_list)[np.where(err_list<np.median(err_list))[0]]

quaternion_med = np.median(quaternion_list,axis=0) # [ 0.02141192, -0.05339036,  0.02547674,  0.98988326]
translation_med = np.median(translation_list,axis=0) # array([ 0.90830587, -0.09662418,  1.21676785])
scale_med= np.median(scale_list) #0.26575331389896006
r = Rot.from_quat([quaternion_med])
test_rot = r.as_dcm()



#r = Rot.from_euler('yxz',[0,0,0],degrees=True) 
#test= r.as_dcm()
test_conv = np.c_[test_rot[0].T, translation_med[:,np.newaxis]] 
test_conv = np.r_[test_conv, np.array([0,0,0,1])[np.newaxis,:]]

'''
left_pcd = read_point_cloud(left_pcd_data[i])
right_pcd = read_point_cloud(right_pcd_data[i])
#right_pcd.transform(test_conv)
points_left = np.asarray(left_pcd.points)
points_right = np.asarray(right_pcd.points)
point_right_convd = points_right.dot(test_rot[0]) + translation_med
diff = points_left - point_right_convd  
#diff = points_left - points_right
'''

np.savetxt("extrinsic_param.csv",test_conv)
