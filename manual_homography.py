import cv2
import numpy as np

 
if __name__ == '__main__' :

    im_src = cv2.imread('prova.jpg')
    pts_src = np.array([[893, 74], [781, 220], [415, 355],[520, 474]])
    im_dst = cv2.imread('prova2.jpg')
    pts_dst = np.array([[861, 79],[770, 203],[456, 319],[544, 421]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey(0)
