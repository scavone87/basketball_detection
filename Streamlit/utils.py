import numpy as np
import cv2 as cv
import os.path
import os
import base64


def load_first_points():
    src_list = np.array( [[230,105], [574, 103], [905, 100], [24, 257], [573, 249], [1103, 243]])  
    dst_list = np.array([[72,27], [586,27], [1100, 27], [72, 324], [586, 324], [1100, 324]])
    return src_list, dst_list


def load_second_points():
    src_list = np.array([[238, 65],[578, 65],[1015, 62] , [78, 192], [579, 223], [1335, 253]])  
    dst_list = np.array([[72,27], [586,27], [1100, 27], [72, 324], [586, 324], [1100, 324]])
    return src_list, dst_list

def draw_points(img, points):
    for point in points:
        x,y = point
        cv.circle(img, (x,y), 5, (0,255,0), 7)
    return img

def get_plan_view(src, dst, fileNameHMatrix, src_list = None, dst_list = None, dim = None):
    if dim is not None: src = cv.resize(src, dim, interpolation = cv.INTER_AREA)
    if os.path.isfile(fileNameHMatrix):
        H = np.loadtxt(fileNameHMatrix).reshape(3, 3)
    else:
        src_pts = np.array(src_list).reshape(-1, 1, 2)
        dst_pts = np.array(dst_list).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        f = open(fileNameHMatrix, "w")
        for row in H:
            np.savetxt(f, row)
        f.close()
    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    return plan_view

def merge_views(src, dst, fileNameHMatrix, src_list = None, dst_list = None, dim = None):
    plan_view = get_plan_view(src, dst, fileNameHMatrix, src_list, dst_list, dim)
    for i in range(0, dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if(plan_view.item(i, j, 0) == 0 and
               plan_view.item(i, j, 1) == 0 and
               plan_view.item(i, j, 2) == 0):
                plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                plan_view.itemset((i, j, 2), dst.item(i, j, 2))
    return plan_view

def find_homography(src, dst, fileNameHMatrix):
     H = np.loadtxt(fileNameHMatrix).reshape(3, 3)
     plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
     for i in range(0, dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if(plan_view.item(i, j, 0) == 0 and
               plan_view.item(i, j, 1) == 0 and
               plan_view.item(i, j, 2) == 0):
                plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                plan_view.itemset((i, j, 2), dst.item(i, j, 2))
     return plan_view

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href