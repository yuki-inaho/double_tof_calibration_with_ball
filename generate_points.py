import pandas as pd
import sys
import re
import os
import glob
import shutil
import pdb

try :
    import cv2
except :
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
    import cv2
import numpy as np
import scipy as sp
from scipy import linalg
from open3d import *
from pycpd import deformable_registration, rigid_registration
import colorsys

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

BALL_NUM = 5

pinhole_camera_intrinsic_right = PinholeCameraIntrinsic(image_width, image_height, fx_right, fy_right, cx_right, cy_right)
pinhole_camera_intrinsic_left = PinholeCameraIntrinsic(image_width, image_height, fx_left, fy_left, cx_left, cy_left)

DATA_DIR_RIGHT = "./data/capture_zense_right"
DATA_DIR_LEFT = "./data/capture_zense_left"

PCD_DIR_LEFT = "./pcd/left"
PCD_DIR_RIGHT  = "./pcd/right"

SPHERE_IMG_DIR_LEFT = "./res_img/left"
SPHERE_IMG_DIR_RIGHT  = "./res_img/right"

def _getColorMask(color_image):
    img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    hsv_color1 = np.asarray([0, 0, 160])   # white!
    hsv_color2 = np.asarray([255, 255, 255])   # yellow! note the order

    mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
    return mask

def getColorMask(color_image):
    mask = _getColorMask(color_image)
    _color_image = np.zeros(color_image.shape)
    _color_image[mask.nonzero()[0], mask.nonzero()[1], :] = color_image[mask.nonzero()[0], mask.nonzero()[1], :]
    masked_image = _color_image.astype(np.uint8)
    return masked_image, mask

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

        #print("{0},{1},{2}".format(r,g,b))
        color_res[np.where(res_mask == (l+1))[0], np.where(res_mask == (l+1))[1],0] = np.floor(r*255)
        color_res[np.where(res_mask == (l+1))[0], np.where(res_mask == (l+1))[1],1] = np.floor(g*255)
        color_res[np.where(res_mask == (l+1))[0], np.where(res_mask == (l+1))[1],2] = np.floor(b*255)

    color_res = color_res.astype(np.uint8)
    return color_res

def MaskSegmentation(color_image, mask):
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask ,cv2.MORPH_OPEN,kernel, iterations = 3)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(color_image, markers)
    color_res = result2ColorImage(markers)
    return color_res, markers

def deriveSphereCoeff(color_image, depth_image, mask, pinhole_camera_intrinsic):
    color_masked = color_image.copy()
    depth_masked = depth_image.copy()
    color_masked[:,:,0][np.where(mask == 0)] =  0
    color_masked[:,:,1][np.where(mask == 0)] =  0
    color_masked[:,:,2][np.where(mask == 0)] =  0
    depth_masked[np.where(mask == 0)] =  0

    color = Image(color_masked)
    depth = Image(depth_masked)
    rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False);
    pcd = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)                                                   
    #pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    pcd = voxel_down_sample(pcd, voxel_size = 0.005)
    points = np.asarray(pcd.points)

    n = points.shape[0]
    r = points[:,0]*points[:,0]+points[:,1]*points[:,1]+points[:,2]*points[:,2]
    Z = np.c_[r, points, np.ones(n)]
    M = Z.transpose().dot(Z)/n  
    P = np.zeros([5,5])
    T = np.zeros([5,5])
    P[4,0] = P[0,4] = -2
    P[1,1] = P[2,2] = P[3,3] = 1
    T[0,0] = 4* M[0,4]
    T[0,1] = T[1,0] = 2 * M[0,3]
    T[0,2] = T[2,0] = 2 * M[0,2]
    T[0,3] = T[3,0] = 2 * M[0,1]
    T[1,1] = T[2,2] = T[3,3] = 1
    H = 2*T-P
    if(np.sum(np.isnan(M))>0 or np.sum(np.isnan(H)) > 0): 
        coeff = np.zeros(4)
        status = False
        return points, coeff, status
    eigvals, eigvecs = linalg.eig(M,H)
    eigvals[np.where(eigvals<0)]  = np.inf
    sort_idx = np.argsort(np.abs(eigvals))
    min_eig_var_idx = sort_idx[0]
    _coeff = eigvecs[:, min_eig_var_idx]
    coeff = np.zeros(4)
    coeff[0] = - _coeff[1]/(2*_coeff[0])
    coeff[1] = - _coeff[2]/(2*_coeff[0])
    coeff[2] = - _coeff[3]/(2*_coeff[0])
    coeff[3] = np.sqrt((_coeff[1]*_coeff[1] + _coeff[2]*_coeff[2] + _coeff[3]*_coeff[3] - 4 * _coeff[0] * _coeff[4])/(4*_coeff[0] * _coeff[0]))
    status = True
    if(np.sum(np.isnan(coeff))>0): status = False
    return points, coeff, status

def getSphereCoeff(color_image, depth_image, mask, pinhole_camera_intrinsic, markers):
    n_label = np.max(markers)
    sphere_max_idx = 0
    sphere_idx = 0
    coeff_mat = np.zeros([n_label,4])
    for l in np.arange(n_label):
        mask = np.zeros(markers.shape)
        mask[np.where(markers == l+1)] = 255
        if np.sum(markers == l+1) > 10000 :
            print("not sphere mask")
            continue

        points, coeff, status = deriveSphereCoeff(color_image, depth_image, mask, pinhole_camera_intrinsic)
        if(not status): continue
        r = coeff[3]
        a = coeff[0]
        b = coeff[1]
        c = coeff[2]

        #print("idx:{0}, radius:{1}, center_x:{2}, center_y:{3}, center_z:{4}".format(l,r,a,b,c))    
        coeff_mat[sphere_idx,0] = r
        coeff_mat[sphere_idx,1] = a
        coeff_mat[sphere_idx,2] = b
        coeff_mat[sphere_idx,3] = c
        sphere_idx = sphere_idx + 1
        sphere_max_idx = sphere_max_idx + 1
    coeff_mat = coeff_mat[np.arange(sphere_max_idx), :]
    return coeff_mat, sphere_max_idx

def drawSphereImg(color_image, coeff_mat, sphere_max_idx, fx, fy, cx, cy):
    coeff_mat = coeff_mat[np.arange(sphere_max_idx), :]
    res_image = np.copy(color_image)
    mean_radius = 0
    for i in np.arange(coeff_mat.shape[0]):
        w = int(np.floor(cx + (coeff_mat[i,1]  * fx ) / coeff_mat[i,3]))
        h = int(np.floor(cy + (coeff_mat[i,2]  * fy ) / coeff_mat[i,3]))
        w2 = int(np.floor(cx + ((coeff_mat[i,1]+coeff_mat[i,0])  * fx ) /coeff_mat[i,3]))
        mean_radius += np.abs(w2-w)/coeff_mat.shape[0]
        res_image = cv2.circle(res_image,(w,h), np.abs(w2-w), (0,0,255), thickness=2)
    return res_image


save_path = PCD_DIR_LEFT
if not os.path.exists(save_path):
    os.mkdir(save_path)
else :
    shutil.rmtree(save_path)
    os.mkdir(save_path)

save_path = PCD_DIR_RIGHT
if not os.path.exists(save_path):
    os.mkdir(save_path)
else :
    shutil.rmtree(save_path)
    os.mkdir(save_path)

save_path = SPHERE_IMG_DIR_RIGHT
if not os.path.exists(save_path):
    os.mkdir(save_path)
else :
    shutil.rmtree(save_path)
    os.mkdir(save_path)

save_path = SPHERE_IMG_DIR_LEFT
if not os.path.exists(save_path):
    os.mkdir(save_path)
else :
    shutil.rmtree(save_path)
    os.mkdir(save_path)


left_color_images = glob.glob(DATA_DIR_LEFT + "/color_*.jpg")
left_color_images = np.sort(left_color_images)
right_color_images = glob.glob(DATA_DIR_RIGHT + "/color_*.jpg")
right_color_images = np.sort(right_color_images)

left_ir_images = glob.glob(DATA_DIR_LEFT + "/ir_*.png")
left_ir_images = np.sort(left_ir_images)
right_ir_images = glob.glob(DATA_DIR_RIGHT + "/ir_*.png")
right_ir_images = np.sort(right_ir_images)

left_depth_images = glob.glob(DATA_DIR_LEFT + "/depth_*.png")
left_depth_images = np.sort(left_depth_images)
right_depth_images = glob.glob(DATA_DIR_RIGHT + "/depth_*.png")
right_depth_images = np.sort(right_depth_images)

sphere_count = 0
i=0
for i in np.arange(len(left_color_images)):
    left_color_img = cv2.imread(left_color_images[i])
    right_color_img = cv2.imread(right_color_images[i])
    left_ir_img = cv2.imread(left_ir_images[i])
    right_ir_img = cv2.imread(right_ir_images[i])
    left_depth_img = cv2.imread(left_depth_images[i],cv2.IMREAD_ANYDEPTH)
    right_depth_img = cv2.imread(right_depth_images[i],cv2.IMREAD_ANYDEPTH)

    right_masked_img, right_mask = getColorMask(right_color_img)
    left_masked_img, left_mask = getColorMask(left_color_img)

    _right_markers_img, right_markers = MaskSegmentation(right_color_img, right_mask)
    _left_markers_img, left_markers = MaskSegmentation(left_color_img, left_mask)

    n_label_right = np.max(right_markers)
    n_label_left = np.max(left_markers)

    left_sphere_center = []
    right_sphere_center = []

    detected = False

    if n_label_left == n_label_right :
        sphere_coeff_left, sphere_max_idx_left = getSphereCoeff(left_color_img, left_depth_img, left_mask, pinhole_camera_intrinsic_left, left_markers)
        sphere_coeff_right, sphere_max_idx_right = getSphereCoeff(right_color_img, right_depth_img, right_mask, pinhole_camera_intrinsic_right, right_markers)

        if(sphere_coeff_left.shape[0] == sphere_coeff_right.shape[0]):
            res_img_left = drawSphereImg(left_color_img, sphere_coeff_left, sphere_max_idx_left, fx_left, fy_left, cx_left, cy_left)
            res_img_right = drawSphereImg(right_color_img, sphere_coeff_right, sphere_max_idx_right, fx_right, fy_right, cx_right, cy_right)
            res_img_name = "/result_%04d"%(sphere_count) + ".png"    
            cv2.imwrite(SPHERE_IMG_DIR_LEFT + res_img_name, res_img_left)
            cv2.imwrite(SPHERE_IMG_DIR_RIGHT + res_img_name, res_img_right)


            if sphere_coeff_left.shape[0] != BALL_NUM or sphere_coeff_right.shape[0] != BALL_NUM:
                continue

            pcd_left = PointCloud()
            pcd_left.points = Vector3dVector(sphere_coeff_left[:,1:])
            pcd_right = PointCloud()
            pcd_right.points = Vector3dVector(sphere_coeff_right[:,1:])

            write_point_cloud(PCD_DIR_LEFT+ "/%04d.pcd"%(sphere_count), pcd_left)
            write_point_cloud(PCD_DIR_RIGHT + "/%04d.pcd"%(sphere_count), pcd_right)
            detected = True
            sphere_count += 1

    if detected:
        print("idx:%d sphere detected!", i)
    else :
        print("idx:%d sphere not detected...", i)


print("sphere_count:%d"%(sphere_count))

'''
key = cv2.waitKey(10)
while key & 0xFF != 27:
    cv2.imshow("test", res_img_right)
    key = cv2.waitKey(10)
cv2.destroyAllWindows()

'''