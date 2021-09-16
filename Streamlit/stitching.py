import streamlit as st
import numpy as np
import cv2 as cv
import utils


def app():
    st.title("Stitching")


    selezione = st.selectbox("Seleziona: ", ["Campo dritto","Campo storto"])