import streamlit as st
import cv2
from constants import PATH_IMGS
import utils



def app():
    
    st.title("Punti di omografia")

    st.write('''Per calcolare la matrice di omografia H è stato utilizzato un approccio manuale. 
    L'obiettivo è quello di ricavare la matrice dalla prima immagine e riutilizzarla per le successive. 
    Il limite di questo approccio è che funziona bene quando l'immagine di query è il quanto più simile all'immagine utilizzata per ricavare la matrice H.
    Il campo di applicazione ideale sarebbe avere un inquadratura statica. Nel basket però le inquadrature sono dinamiche (pan, zoom ...).
    Nel nostro caso però, andando a fare uno stitching per ottenere un'immagine panoramica di tutto il campo, abbiamo notato come otteniamo due inquadrature che si ripetono.
    Un'inquadratura che tende ad avere i lati quasi perfettamente simmetrici (Visuale 1) e un'inquadratura che tende ad avere il lato destro più lungo del sinistro (Visuale 2).
    Abbiamo deciso dunque di catalogare le immagini in base a queste due visuali e utilizzare a seconda di esse la matrice di omografia adatta.
    ''')

    st.header("**Visuale 2**: Lato destro più lungo del sinistro")

    src2 = cv2.imread(PATH_IMGS + "/src2.jpg")
    dst = cv2.imread(PATH_IMGS + "/dst.jpg")

    src_points, dst_points = utils.load_first_point()
    
    im_src = utils.draw_points(src2.copy(), src_points)
    im_dst = utils.draw_points(dst.copy(), dst_points)
    
    st.image(im_src, channels='BGR', caption="Query Image")
    st.image(im_dst, channels= 'BGR', caption="Train Image")

    im_out = utils.merge_views(src2, dst, src_points, dst_points, "homography_1")

    st.image(im_out, channels='BGR', caption="Homography with H matrix")

    

