import streamlit as st
import cv2
from streamlit import util
import utils

PATH_IMGS = "imgs/"

def app():
    
    st.title("Punti di omografia")

    st.write(
    '''Per calcolare la matrice di omografia H è stato utilizzato un approccio
       manuale. L'obiettivo è quello di ricavare la matrice dalla prima immagine e riutilizzarla per le successive 
    ''')

    src = cv2.imread(PATH_IMGS + "/src.jpg")
    dst = cv2.imread(PATH_IMGS + "/dst.jpg")

    src_points, dst_points = utils.load_first_point()
    
    im_src = utils.draw_points(src.copy(), src_points)
    im_dst = utils.draw_points(dst.copy(), dst_points)
    
    st.image(im_src, channels='BGR', caption="Query Image")
    st.image(im_dst, channels= 'BGR', caption="Train Image")

    im_out = utils.merge_views(src, dst, src_points, dst_points, "homography_1")

    st.image(im_out, channels='BGR', caption="Homography with H matrix")

    

