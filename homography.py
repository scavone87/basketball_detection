import numpy as np
import cv2 as cv
import tkinter as tk

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

img_list = [ [159,372], [783,218], [235,496], [1091,430] ]
img_list1 = [[236,334],[772,203],[453,316],[1025,383]]
img_list2 = [[297,346],[822,207],[603,431],[1095,398]]

dst_list = [ [488,138], [681,15], [489,254], [679,255] ]
dst_list1 = [ [488,137], [680,15], [541,137], [680,254] ]
dst_list2 = [ [488,138], [680,16], [541,255], [678,253] ]


def selected_points(img, point_list):
    for x,y in point_list:
        cv.circle(img,(x,y),8,(0,255,0),-1)

def get_plan_view(img, dst, image_order):
    img_pts = []
    dst_pts = []
    if image_order == 1:
        img_pts = np.array(img_list).reshape(-1,1,2)
        dst_pts = np.array(dst_list).reshape(-1,1,2)
    elif image_order == 2:
        img_pts = np.array(img_list1).reshape(-1,1,2)
        dst_pts = np.array(dst_list1).reshape(-1,1,2)
    else:
        img_pts = np.array(img_list2).reshape(-1,1,2)
        dst_pts = np.array(dst_list2).reshape(-1,1,2)
    H, mask = cv.findHomography(img_pts, dst_pts, cv.RANSAC,5.0)
    print("H:")
    print(H)
    plan_view = cv.warpPerspective(img, H, (dst.shape[1], dst.shape[0]))
    return plan_view

def merge_views(img, dst, image_order):
    plan_view = get_plan_view(img, dst, image_order)
    for i in range(0,dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if(plan_view.item(i,j,0) == 0 and \
               plan_view.item(i,j,1) == 0 and \
               plan_view.item(i,j,2) == 0):
                plan_view.itemset((i,j,0),dst.item(i,j,0))
                plan_view.itemset((i,j,1),dst.item(i,j,1))
                plan_view.itemset((i,j,2),dst.item(i,j,2))
    return plan_view

def setup_window(path, namedWindow, x, y):
    img = cv.imread(path, -1)    
    cv.namedWindow(namedWindow)
    cv.moveWindow(namedWindow, x,y)
    return img

#source1
img = setup_window('immagini/img.jpg', 'img', 0, 0)
img_copy = img.copy()

#source2
img1 = setup_window('immagini/img1.jpg', 'img1', 0, int(screen_height/2))
img_copy1 = img1.copy()

#source3
img2 = setup_window('immagini/img2.jpg', 'img2', int(screen_width/2), 0)
img_copy2 = img2.copy()

dst = setup_window('immagini/dst.jpg', 'dst', int(screen_width/2), int(screen_height/2))
dst_copy = dst.copy()


while(1):
    cv.imshow('img',img_copy)
    selected_points(img_copy, img_list)

    cv.imshow('img1', img_copy1)
    selected_points(img_copy1, img_list1)

    cv.imshow('img2', img_copy2)
    selected_points(img_copy2, img_list2)

    cv.imshow('dst',dst_copy)
    k = cv.waitKey(1) & 0xFF
    if k == ord('1'):
        dst = setup_window('immagini/dst.jpg', 'dst', int(screen_width/2), int(screen_height/2))
        dst_copy = dst.copy()
        merge = merge_views(img, dst, 1)
        selected_points(dst_copy, dst_list)
        cv.imshow("merge", merge)
        cv.moveWindow("merge", int(screen_width/2)-350,int(screen_height/2)-212)

    elif k == ord("2"):
        dst = setup_window('immagini/dst.jpg', 'dst', int(screen_width/2), int(screen_height/2))
        dst_copy = dst.copy()
        merge = merge_views(img1, dst, 2)
        selected_points(dst_copy, dst_list1)
        cv.imshow("merge", merge)

    elif k == ord("3"):
        dst = setup_window('immagini/dst.jpg', 'dst', int(screen_width/2), int(screen_height/2))
        dst_copy = dst.copy()
        merge = merge_views(img2, dst, 3)
        selected_points(dst_copy, dst_list2)
        cv.imshow("merge", merge)        
    elif k == ord('q'):
        break
cv.destroyAllWindows()
