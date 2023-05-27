import streamlit as st
import numpy as np
import cv2
from constants import PATH_IMGS_EXAMPLE, PATH_VIDEO_EXAMPLE, FRAME_EXTRACTOR, PATH_SCRIPTS
import utils



st.set_page_config(page_title='Basketball Homography', page_icon=':basketball:', layout='centered', initial_sidebar_state='auto')   

#st.sidebar.title('Navigazione')
#st.sidebar.success("Seleziona una pagina")

st.title("Introduzione")
st.subheader("Esempi di video utilizzati")
col1,col2 = st.columns(2)

video_file_lakers = open(PATH_VIDEO_EXAMPLE + 'lakers.mp4', 'rb') 
video_bytes_lakers = video_file_lakers.read()
video_file_oklahoma = open(PATH_VIDEO_EXAMPLE + 'oklahoma.mp4', 'rb') 
video_bytes_oklahoma = video_file_oklahoma.read()

col1.video(video_bytes_lakers)
col2.video(video_bytes_oklahoma)
    
st.markdown('''<div style="text-align: justify">
Per questo progetto sono stati utilizzati frame estratti dal videogame NBA2K21 oltre a immagini provenienti da partite reali, entrambi ricavati mediante uno script
python, che permette di ottenere un frame per ogni secondo della clip. </div>
<br>
Di seguito il codice:
''', unsafe_allow_html= True)

my_expander = st.expander(label='Click to show code')
with my_expander:
    st.code(FRAME_EXTRACTOR, language= 'python')
    st.write("**NOTA**: Il ritardo con cui estrapolare i frame può essere modificato opportunamente alla seguente riga di codice `vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*delay_in_milliseconds))`")
    st.markdown(utils.get_binary_file_downloader_html(PATH_SCRIPTS + 'frame_extractor.py', 'Code'), unsafe_allow_html=True)

    real = cv2.imread(PATH_IMGS_EXAMPLE + "/real.jpg")
    game = cv2.imread(PATH_IMGS_EXAMPLE + "/game.jpg")
    real1 = cv2.imread(PATH_IMGS_EXAMPLE + "/real1.jpg")
    game1 = cv2.imread(PATH_IMGS_EXAMPLE + "/game1.jpg")
    real2 = cv2.imread(PATH_IMGS_EXAMPLE + "/real2.png")
    game2 = cv2.imread(PATH_IMGS_EXAMPLE + "/game2.jpg")

    st.subheader("Alcuni frame estratti")
    st.image(real, channels='BGR', caption="TD Garden (Boston)")
    st.image(real1, channels= 'BGR', caption="Staples Center - Lakers (Los Angeles)")
    st.image(real2, channels='BGR', caption="Mediolanum Forum (Milano)")
    st.image(game, channels= 'BGR', caption="Staples Center - Lakers (Los Angeles) da NBA2K21")
    st.image(game1, channels='BGR', caption="Chesapeake Energy Arena (Oklahoma) da NBA2K21")
    st.image(game2, channels= 'BGR', caption="Scotiabank Arena (Toronto) da NBA2K21")

    st.markdown("---")

    st.subheader("Nota")
    st.write(
    '''Risulta chiara, dagli esempi sopra presentati, la presenza di watermark.
        Questi ultimi sono stati rimossi in una fase di pre-elaborazione delle immagini per garantire una migliore riuscita dello stitching.
    ''')

    watermark = cv2.imread(PATH_IMGS_EXAMPLE + "/watermark.jpg")
    nowatermark = cv2.imread(PATH_IMGS_EXAMPLE + "/nowatermark.jpg")

    st.image(watermark, channels='BGR', caption="Immagine con watermark")
    st.image(nowatermark, channels= 'BGR', caption="Immagine senza watermark")