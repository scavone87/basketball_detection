PATH_IMGS_EXAMPLE = "introduction/"
PATH_VIDEO_EXAMPLE = "introduction/videos/"
PATH_SCRIPTS = "script_utili/"
PATH_LA = "stitching/Lakers/"
PATH_OKC = "stitching/Oklahoma/"
PATH_IMGS = "homography/imgs/"

PATH_VISUALE_1 = "homography/visuale1"
PATH_VISUALE_2 = "homography/visuale2"


H_MATRIX_1 = "homography_1.txt"
H_MATRIX_2 = "homography_2.txt"


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
% Pulizia del workspace e delle variabili
clear
clc

% Caricamento delle immagini
buildingDir = fullfile('Inserire il path della cartella contenente i frame per lo stitching');
buildingScene = imageDatastore(buildingDir);

% Stampa delle immagini che verranno usate per lo stitching
%montage(buildingScene.Files)

% Lettura della prima immagine dell'image set
I = readimage(buildingScene,1);

% Inizializzazione delle features
grayImage = im2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage,points);

% Inizializza tutte le trasformazioni alla matrice identità
numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));

% Inizializzazione delle variabili per mantenere le dimensioni
% dell'immagine
imageSize = zeros(numImages,2);

% Iterazione, a coppie di due, sulle immagini restanti
for n = 2:numImages
    
    % Memorizzazione dei punti e delle features
    pointsPrevious = points;
    featuresPrevious = features;
        
    % Lettura dell'immagine n
    I = readimage(buildingScene, n);
    
    % Conversione dell'immagine in scala di grigi
    grayImage = im2gray(I);    
    
    % Salvataggio delle misure dell'immagine
    imageSize(n,:) = size(grayImage);
    
    % Riconoscimento ed estrazione delle features per applicare SURF all'immagine n
    points = detectSURFFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);
  
    % Ricerca delle corrispondenze tra l'immagine n e l'immagine n-1
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
    
    % Stima della trasformazione tra l'immagine n e l'immagine n-1
    tforms(n) = estimateGeometricTransform2D(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    % Calcolo T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

% Calcolo dei bordi per ogni trasformazione in output
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

% Ricerca del minimo e del massimo inerente ai bordi dell'output 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Larghezza e altezza dell'immagine panoramica ottenuta
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Inizializzazione dell'immagine panoramica "vuota"
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Creazione di un oggetto 2D con le misure prima definite
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Creazione dell'immagine panoramica
for i = 1:numImages
    
    I = readimage(buildingScene, i);   
   
    % Trasformazione dell'immagine i nell'immagine panoramica
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Creazione di una maschera binaria    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Sovrapposizione dell'immagine elaborata al panorama
    panorama = step(blender, panorama, warpedImage, mask);
end

figure('Name','Immagine paronamica ottenuta','NumberTitle','off', 'Color','black');
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