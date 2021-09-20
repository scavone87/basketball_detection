PATH_IMGS_EXAMPLE = "introduction/"
PATH_VIDEO_EXAMPLE = "introduction/videos/"
PATH_SCRIPTS = "script_utili/"
PATH_LA = "stitching/Lakers/"
PATH_OKC = "stitching/Oklahoma/"
PATH_IMGS = "homography/imgs/"


FRAME_EXTRACTOR = '''
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
'''

MATLAB_STITCHING = '''
clear;      
clc;

% Load images.
buildingDir = fullfile('Inserire il path della cartella contenente i frame per lo stitching');
buildingScene = imageDatastore(buildingDir);

% Display images to be stitched.
montage(buildingScene.Files)

% Read the first image from the image set.
I = readimage(buildingScene,1);

% Initialize features for I(1)
grayImage = im2bw(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage,points);

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages

    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
	        
    % Read I(n).
    I = readimage(buildingScene, n);
	    
    % Convert image to grayscale.
    grayImage = im2bw(I);    
	    
    % Save image size.
    imageSize(n,:) = size(grayImage);
    
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);
	  
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
	       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
	    
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
	    
    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

% Compute the output limits for each transform.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
   	tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = readimage(buildingScene, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
	                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
	    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)

'''

PYTHON_STITCHING = '''
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# costruzione parser argomenti e analizza gli argomenti
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input directory of input to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
ap.add_argument("-t", "--type", type=str, default="d",
	help="type of the input, 'd' is directory, 'v' is video")
args = vars(ap.parse_args())
# prendere i percorsi delle immagini di input e inizializzare la nostra lista di immagini
images = []

if args["type"] == "d":
  print("[INFO] loading images...")
  imagePaths = sorted(list(paths.list_images(args["input"])))
  # loop sui percorsi delle immagini
  # caricare ognuna di esse e aggiungerle alla nostra lista di immagini da cucire
  for imagePath in imagePaths:
  	image = cv2.imread(imagePath)
  	images.append(image)
  # inizializza l'oggetto image sticher di OpenCV 
  # poi esegue lo stitching dell'immagine
  print("[INFO] stitching images...")
elif args["type"] =="v":
  cap=cv2.VideoCapture(args["input"])
  Num = 0
  while Num < cap.get(cv2.CAP_PROP_FRAME_COUNT):
     Num += 1
     if Num % 1 == 0:
       ret,frame=cap.read()
       images.append(frame)

stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# Se lo status è '0', 
# allora OpenCV ha eseguito con successo la cucitura dell'immagine
if status == 0:
	cv2.imwrite(args["output"], stitched)
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
# Altrimenti lo stitching è fallito
# probabilmente a causa di un numero insufficiente di punti chiave
else:
	print("[INFO] image stitching failed ({})".format(status))'''