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
from checkerboard import detect_checkerboard

fx_left = 627.601
fy_left = 628.045
cx_left = 233.937
cy_left = 340.018

fx_right = 460
fy_right = 460.716
cx_right = 240.645
cy_right = 335.792

image_width = 480
image_height = 640

TEST_DIR_LEFT = "./data/capture_zense_left"
TEST_DIR_RIGHT = "./data/capture_zense_right"

left_ir_images = glob.glob(TEST_DIR_LEFT + "/ir_*.png")
left_ir_images = np.sort(left_ir_images)
right_ir_images = glob.glob(TEST_DIR_RIGHT + "/ir_*.png")
right_ir_images = np.sort(right_ir_images)

left_depth_images = glob.glob(TEST_DIR_LEFT + "/depth_*.png")
left_depth_images = np.sort(left_depth_images)
right_depth_images = glob.glob(TEST_DIR_RIGHT + "/depth_*.png")
right_depth_images = np.sort(right_depth_images)

left_depth_color_images = glob.glob(TEST_DIR_LEFT + "/depth_color_*.jpg")
left_depth_color_images = np.sort(left_depth_color_images)
right_depth_color_images = glob.glob(TEST_DIR_RIGHT + "/depth_color_*.jpg")
right_depth_color_images = np.sort(right_depth_color_images)


test_conv = np.loadtxt("extrinsic_param.csv")
#test_conv[:3,3] = np.array([0.11160231, 0.01174368, 0.02080043])

i=8
left_ir_img = cv2.imread(left_ir_images[i])
right_ir_img = cv2.imread(right_ir_images[i])
left_depth_img = cv2.imread(left_depth_images[i],cv2.IMREAD_ANYDEPTH)
right_depth_img = cv2.imread(right_depth_images[i],cv2.IMREAD_ANYDEPTH)
left_depth_color_img = cv2.imread(left_depth_color_images[i])
right_depth_color_img = cv2.imread(right_depth_color_images[i])

color_left_o3d = Image(left_depth_color_img)
depth_left_o3d = Image(left_depth_img)

pinhole_camera_intrinsic_left = PinholeCameraIntrinsic(image_width, image_height, fx_left, fy_left, cx_left, cy_left)
rgbd_left = create_rgbd_image_from_color_and_depth(color_left_o3d, depth_left_o3d, convert_rgb_to_intensity = False)
pcd_left = create_point_cloud_from_rgbd_image(rgbd_left, pinhole_camera_intrinsic_left)                                                   
pcd_left.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
#pcd_left.transform([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

pinhole_camera_intrinsic_right = PinholeCameraIntrinsic(image_width, image_height, fx_right, fy_right, cx_right, cy_right)
color_right_o3d = Image(right_depth_color_img)
depth_right_o3d = Image(right_depth_img)
rgbd_right = create_rgbd_image_from_color_and_depth(color_right_o3d, depth_right_o3d, convert_rgb_to_intensity = False)
pcd_right = create_point_cloud_from_rgbd_image(rgbd_right, pinhole_camera_intrinsic_right)   
pcd_right.transform(test_conv)
pcd_right.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

draw_geometries([pcd_left,pcd_right])

for y in np.arange(3):
    for x in np.arange(4):
        print("tf%d%d=%f"%(y,x,test_conv[y,x]))