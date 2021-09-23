import streamlit as st
import cv2
from constants import PATH_IMGS, H_MATRIX_1, H_MATRIX_2, PATH_VISUALE_1, PATH_VISUALE_2
import utils
import glob

visual_dict = {
    'Visuale 1': PATH_VISUALE_1,
    'Visuale 2': PATH_VISUALE_2
}


def app():

    st.title("Punti di omografia")

    st.markdown('''<div style="text-align: justify"> Per calcolare la matrice di omografia H è stato utilizzato un approccio manuale. 
    L'obiettivo è quello di ricavare la matrice dalla prima immagine e riutilizzarla per le successive. 
    Il limite di questo approccio è che funziona bene quando l'immagine di source è il quanto più simile all'immagine utilizzata per ricavare la matrice H.
    Il campo di applicazione ideale è quello in cui le immagini vengono ricavate da un'inquadratura statica. Nel basket però le inquadrature sono dinamiche (pan, zoom ...).</div>''', unsafe_allow_html=True)

    
    st.markdown('''<br><div style="text-align: justify"> Nel nostro caso, facendo uno stitching per ottenere un'immagine panoramica di tutto il campo, si è notato che si ottengono due inquadrature che si ripetono: 
    una tende ad avere i lati lunghi del campo quasi perfettamente paralleli, rendendo le due metà campo simmetriche rispetto al centrocampo (<b>Visuale 1</b>), 
    l'altra tende ad avere il lato destro del campo più lungo del sinistro, mostrando l'area destra più grande rispetto alla sinistra (<b>Visuale 2</b>).</div>
    <br> <div style="text-align: justify"> Abbiamo deciso dunque di catalogare le immagini in base a queste due visuali e utilizzare a seconda di esse la matrice di omografia adatta.</div>
    ''', unsafe_allow_html=True)

    st.header("**Visuale 1**")

    src1 = cv2.imread(PATH_IMGS + "/src1.jpg")
    dst = cv2.imread(PATH_IMGS + "/dst.jpg")

    src1_points, dst_points = utils.load_first_points()
    im_src1 = utils.draw_points(src1.copy(), src1_points)
    im_dst = utils.draw_points(dst.copy(), dst_points)

    st.image(im_src1, channels='BGR', caption="Source Image")
    st.image(im_dst, channels='BGR', caption="Destination Image")

    im_out1 = utils.merge_views(src1, dst, H_MATRIX_1, src1_points, dst_points)
    st.image(im_out1, channels='BGR', caption="Homography with H matrix")

    st.markdown("---")

    st.header("**Visuale 2**")

    src2 = cv2.imread(PATH_IMGS + "/src2.jpg")

    src2_points, dst_points = utils.load_second_points()

    im_src2 = utils.draw_points(src2.copy(), src2_points)

    st.image(im_src2, channels='BGR', caption="Source Image")
    st.image(im_dst, channels='BGR', caption="Destination Image")

    im_out2 = utils.merge_views(src2, dst, H_MATRIX_2, src2_points, dst_points)
    st.image(im_out2, channels='BGR', caption="Homography with H matrix")

    st.markdown("---")

    current_visual = st.selectbox("Seleziona una visuale", visual_dict.keys())
    st.markdown("<br>", unsafe_allow_html=True)
   

    current_src = []
    if current_visual == 'Visuale 1':
        current_src = src1
        H_file = H_MATRIX_1
    else:
        current_src = src2
        H_file = H_MATRIX_2

    images = [cv2.imread(file) for file in glob.glob(
        visual_dict[current_visual]+"/*.jpg")]
    i = 1
    for image in images:
        expander = st.expander(label=f'Esempio #{i}')
        with expander:
            st.image(image, channels='BGR', caption="Source image")
            result_original_dim = utils.merge_views(image, dst, H_file)
            st.image(result_original_dim, channels='BGR',
                     caption="Homography with original dimension")
            result = utils.merge_views(image, dst, fileNameHMatrix=H_file, dim=(
                current_src.shape[1], current_src.shape[0]))
            st.image(result, channels='BGR',
                     caption="Homography with same dimension of original src")
        i = i+1
