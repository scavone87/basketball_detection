import streamlit as st
import numpy as np
import imageio
import imutils
import cv2
from constants import PATH_IMGS

feature_extractors = ['sift', 'surf', 'brisk', 'orb']
feature_matchings = ['knn', 'bf']


def app():

    st.title("Omografia interattiva")

    feature_extractor = st.selectbox(
        "Seleziona l'algoritmo di feature extractor",
        feature_extractors
    )

    feature_matching = st.selectbox(
        "Seleziona l'algoritmo di feature matching",
        feature_matchings
    )

    trainImg = cv2.imread(PATH_IMGS + 'src1.jpg')
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
    queryImg = cv2.imread(PATH_IMGS + 'dst.jpg')
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

    st.image(trainImg, channels="BGR",
             caption="Train Image (Image to be transformed)")
    st.image(queryImg, channels="BGR", caption="Query Image")

    kpsA, featuresA = detectAndDescribe(
        trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(
        queryImg_gray, method=feature_extractor)

    st.image(cv2.drawKeypoints(trainImg_gray, kpsA, None, color=(0, 255, 0)), caption="Key points of train image")
    st.image(cv2.drawKeypoints(queryImg_gray, kpsB, None, color=(0, 255, 0)), caption="Key points of query image")

    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        ratio = st.slider("Lowe Ratio Test", min_value=0.65, max_value= 1.0, value=0.65 ,step= 0.05)
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=ratio, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,np.random.choice(matches,100),None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    st.image(img3, channels="BGR", caption="First 100 matching of key points")

    reprojThresh = st.slider("Ransac Reprojection threshold", min_value=1, max_value=100, step=1)
    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=reprojThresh)
    if M is None:
        st.error("Errore! Numero di matching non sufficienti per ricavare la matrice H")
    (matches, H, status) = M
    st.subheader("Homography Matrix")
    st.table(H)

    plan_view = cv2.warpPerspective(trainImg, H, (queryImg.shape[1], queryImg.shape[0]))
    for i in range(0,queryImg.shape[0]):
        for j in range(0, queryImg.shape[1]):
            if plan_view.item(i,j,0) == 0 and plan_view.item(i,j,1) == 0 and plan_view.item(i,j,2) == 0:
                plan_view.itemset((i,j,0),queryImg.item(i,j,0))
                plan_view.itemset((i,j,1),queryImg.item(i,j,1))
                plan_view.itemset((i,j,2),queryImg.item(i,j,2))

    st.image(plan_view, channels="BGR", caption = "Homography")


def detectAndDescribe(image, method=None):
    """
    Calcola i keypoints e i descrittori delle features usando un metodo specifico
    """

    assert method is not None, "È necessario definire un metodo di rilevamento delle features. I valori sono: 'sift', 'surf'"

    # rilevare ed estrarre le caratteristiche dall'immagine
    if method == 'sift':
        descriptor=cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor=cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor=cv2.BRISK_create()
    elif method == 'orb':
        descriptor=cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features)=descriptor.detectAndCompute(image, None)

    return (kps, features)

def createMatcher(method,crossCheck):
    "Crea e restituisce un oggetto Matcher"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Ordina le features in ordine di distanza.
    # I punti con poca distanza (più similarity) sono ordinati per primi nel vettore.
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # calcola le corrispondenze grezze e inizializza l'elenco delle corrispondenze effettive
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    for m,n in rawMatches:
        # assicurarsi che la distanza sia entro un certo rapporto
        # ratio compreso tra 0 e 1
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None