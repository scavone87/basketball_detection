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
	print("[INFO] image stitching failed ({})".format(status))