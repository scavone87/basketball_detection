import numpy as np
import cv2 as cv


def load_first_point():
    src_list = np.array([[238, 65],[578, 65],[1015, 62] , [78, 192], [579, 223], [1335, 253]])  
    dst_list = np.array([[72,27], [586,27], [1100, 27], [72, 324], [586, 324], [1100, 324]])
    return src_list, dst_list

def draw_points(img, points):
    for point in points:
        x,y = point
        cv.circle(img, (x,y), 5, (0,255,0), 7)
    return img

def get_plan_view(src, dst, src_list, dst_list):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print("H:")
    print(H)
    f = open('homography.txt', 'w')
    f.write("Final homography: \n" + str(H)+"\n")
    f.close()
    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    return plan_view


def merge_views(src, dst, src_list, dst_list):
    plan_view = get_plan_view(src, dst, src_list, dst_list)
    for i in range(0, dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if(plan_view.item(i, j, 0) == 0 and
               plan_view.item(i, j, 1) == 0 and
               plan_view.item(i, j, 2) == 0):
                plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                plan_view.itemset((i, j, 2), dst.item(i, j, 2))
    return plan_view
