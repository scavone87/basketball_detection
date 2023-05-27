import streamlit as st
import numpy as np
import imageio
import imutils
import glob
import cv2
from image_processing import detectAndDescribe, createMatcher, matchKeyPointsBF, matchKeyPointsKNN, getHomography
from constants import PATH_IMGS

feature_extractors = ['sift', 'brisk', 'orb']
feature_matchings = ['knn', 'bf']

src_dict = {
    'Lakers 1': PATH_IMGS + 'src1.jpg',
    'Lakers 2': PATH_IMGS + 'src2.jpg',
    'Oklahoma 1': PATH_IMGS + 'src3.jpg',
    'Oklahoma 2': PATH_IMGS + 'src4.jpg'
}

dst_dict = {
    'Lakers 2D': PATH_IMGS + 'dst.jpg',
    'Oklahoma 2D': PATH_IMGS + 'dst2.jpg'
}

st.set_page_config(page_title='Omografia interattiva', page_icon=':basketball:', layout='centered', initial_sidebar_state='auto')   

st.title("Omografia interattiva")

st.markdown('''
<div style="text-align: justify"> La seguente pagina consente di applicare gli algoritmi di feature extractor e di feature matching per ottenere una mappatura sul campo 2D
attraverso il calcolo della matrice di omografia. Questo modulo vuole dimostrare che i classici algoritmi possono funzionare bene quando l'immagine di source è molto
simile se non uguale all'immagine di destinazione.</div>
<br>
<div style="text-align: justify"> Con le immagini <b>Lakers 1</b> e <b>Lakers 2D</b>, utilizzando <b>sift</b> con <b>bf</b> e scegliendo come <b>Reprojection threshold</b> 10
si ottiene un buon risultato. </div><br>
''', unsafe_allow_html= True)

current_src = st.selectbox("Seleziona un'immagine di source", src_dict.keys())
st.markdown("<br>", unsafe_allow_html=True)

current_dst = st.selectbox("Seleziona un'immagine di destination", dst_dict.keys())
st.markdown("<br>", unsafe_allow_html=True)

feature_extractor = st.selectbox(
    "Seleziona l'algoritmo di feature extractor",
    feature_extractors
)

feature_matching = st.selectbox(
    "Seleziona l'algoritmo di feature matching",
    feature_matchings
)

if current_dst == 'Lakers 2D':
    queryImg = cv2.imread(PATH_IMGS + 'dst.jpg')
else:
    queryImg = cv2.imread(PATH_IMGS + 'dst2.jpg')

trainImg = cv2.imread(src_dict[current_src])
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
queryImg = cv2.imread(dst_dict[current_dst])
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

st.image(trainImg, channels="BGR",
            caption="Train Image (Image to be transformed)")
st.image(queryImg, channels="BGR", caption="Query Image")

kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

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

reprojThresh = st.slider("Ransac Reprojection threshold", min_value=1.0, max_value=10.0, step=0.5)
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
