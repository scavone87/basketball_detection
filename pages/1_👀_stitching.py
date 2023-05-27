import streamlit as st
import numpy as np
import cv2
from constants import PATH_LA, PATH_OKC, MATLAB_STITCHING, PATH_SCRIPTS, PYTHON_STITCHING
import utils


st.set_page_config(page_title='Stitching', page_icon=':basketball:', layout='centered', initial_sidebar_state='auto')   

st.title("Stitching")

st.markdown('''<div style="text-align: justify">
Al fine di ottenere l'omografia di un intero campo da gioco, i frames estratti sono stati uniti grazie alla
tecnica dello 'stitching'. Per effettuare lo stitching sono stati sviluppati due script: uno utilizzando Python con OpenCV e
l'altro sviluppato in MATLAB.</div> <br>
''', unsafe_allow_html= True)

matlab_section = st.expander(label='MATLAB')
with matlab_section:
    st.code(MATLAB_STITCHING, language='MATLAB')
    st.markdown(utils.get_binary_file_downloader_html(
        PATH_SCRIPTS + 'stitching_matlab.m', 'Code'), unsafe_allow_html=True)

python_section = st.expander(label='Python')
with python_section:
    st.code(PYTHON_STITCHING, language='python')
    st.markdown(utils.get_binary_file_downloader_html(
        PATH_SCRIPTS + 'stitch_multiple_images.py', 'Code'), unsafe_allow_html=True)

st.markdown("---")

st.write(
    '''Di seguito due esempi di stitching, con immagini estratte dal videogioco NBA2K21.
''')

st.subheader("ESEMPIO 1: Staples Center Lakers senza giocatori")

framesx = cv2.imread(PATH_LA + "/1.jpg")
framec = cv2.imread(PATH_LA + "/2.jpg")
framedx = cv2.imread(PATH_LA + "/3.jpg")
result_mat = cv2.imread(PATH_LA + "/res.jpg")
result_py = cv2.imread(PATH_LA + "/res_py.jpg")

st.image(framesx, channels='BGR', caption="Frame sx")
st.image(framec, channels='BGR', caption="Frame centrale")
st.image(framedx, channels='BGR', caption="Frame dx")
st.image(result_py, channels='BGR',
         caption="Risultato dello stitching in Python")
st.image(result_mat, channels='BGR',
         caption="Risultato dello stitching in MATLAB")

st.markdown("---")

st.subheader("ESEMPIO 2: Chesapeake Energy Arena Oklahoma con giocatori")

framesx1 = cv2.imread(PATH_OKC + "/1.jpg")
framec1 = cv2.imread(PATH_OKC + "/2.jpg")
framedx1 = cv2.imread(PATH_OKC + "/3.jpg")
result1_mat = cv2.imread(PATH_OKC + "/res.jpg")
result1_py = cv2.imread(PATH_OKC + "/res_py.jpg")

st.image(framesx1, channels='BGR', caption="Frame sx")
st.image(framec1, channels='BGR', caption="Frame centrale")
st.image(framedx1, channels='BGR', caption="Frame dx")
st.image(result1_py, channels='BGR', caption="Risultato dello stitching in Python")
st.image(result1_mat, channels='BGR',caption="Risultato dello stitching in MATLAB")

st.markdown("---")

st.markdown('''<div style="text-align: justify">
Come è possibile vedere dai seguenti risulati,  <b>MATLAB</b> produce un'immagine meno distorta ed in generale <b>migliore</b>. 
Perciò da questo punto in avanti sono stati utilizzati solo stitching prodotti dallo script MATLAB </div>''', unsafe_allow_html= True)
