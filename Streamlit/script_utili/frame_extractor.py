import sys
import argparse
import cv2

print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture('Insert here filename')
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite( "Insert here output path" + "\\%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="Insert here input path")
    a.add_argument("--pathOut", help="Insert here output path")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)