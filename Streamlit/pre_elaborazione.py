import streamlit as st
import cv2
from streamlit import util
import utils

PATH_IMGS = "imgs/"

def app():
    
    st.title("Punti di omografia")

    src = cv2.imread(PATH_IMGS + "/src.jpg")
    dst = cv2.imread(PATH_IMGS + "/dst.jpg")

    src_points, dst_points = utils.load_first_point()
    
    im_src = utils.draw_points(src.copy(), src_points)
    im_dst = utils.draw_points(dst.copy(), dst_points)
    
    st.image(im_src, channels='BGR')
    st.image(im_dst, channels= 'BGR')

    im_out = utils.merge_views(src, dst, src_points, dst_points)

    st.image(im_out, channels='BGR')

    

