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

    st.write('''Per calcolare la matrice di omografia H è stato utilizzato un approccio manuale. 
    L'obiettivo è quello di ricavare la matrice dalla prima immagine e riutilizzarla per le successive. 
    Il limite di questo approccio è che funziona bene quando l'immagine di query è il quanto più simile all'immagine utilizzata per ricavare la matrice H.
    Il campo di applicazione ideale sarebbe avere un inquadratura statica. Nel basket però le inquadrature sono dinamiche (pan, zoom ...).
    Nel nostro caso però, andando a fare uno stitching per ottenere un'immagine panoramica di tutto il campo, abbiamo notato come otteniamo due inquadrature che si ripetono.
    Un'inquadratura che tende ad avere i lati quasi perfettamente simmetrici (Visuale 1) e un'inquadratura che tende ad avere il lato destro più lungo del sinistro (Visuale 2).
    Abbiamo deciso dunque di catalogare le immagini in base a queste due visuali e utilizzare a seconda di esse la matrice di omografia adatta.
    ''')

    st.header("**Visuale 1**: Lati quasi perfettamente simmetrici")

    src1 = cv2.imread(PATH_IMGS + "/src1.jpg")
    dst = cv2.imread(PATH_IMGS + "/dst.jpg")

    src1_points, dst_points = utils.load_first_points()
    im_src1 = utils.draw_points(src1.copy(), src1_points)
    im_dst = utils.draw_points(dst.copy(), dst_points)

    st.image(im_src1, channels='BGR', caption="Query Image")
    st.image(im_dst, channels= 'BGR', caption="Train Image")

    im_out1 = utils.merge_views(src1, dst, H_MATRIX_1, src1_points, dst_points)
    st.image(im_out1, channels='BGR', caption="Homography with H matrix")

    st.markdown("---")

    st.header("**Visuale 2**: Lato destro più lungo del sinistro")

    src2 = cv2.imread(PATH_IMGS + "/src2.jpg")

    src2_points, dst_points = utils.load_second_points()
    
    im_src2 = utils.draw_points(src2.copy(), src2_points)
    
    st.image(im_src2, channels='BGR', caption="Query Image")
    st.image(im_dst, channels= 'BGR', caption="Train Image")

    im_out2 = utils.merge_views(src2, dst, H_MATRIX_2, src2_points, dst_points)
    st.image(im_out2, channels='BGR', caption="Homography with H matrix")

    current_visual = st.selectbox("Seleziona una visuale", visual_dict.keys())

    current_src = []
    if current_visual == 'Visuale 1':
        current_src = src1
        H_file = H_MATRIX_1
    else:
        current_src = src2
        H_file = H_MATRIX_2

    images = [cv2.imread(file) for file in glob.glob(visual_dict[current_visual]+"/*.jpg")]
    i = 1
    for image in images:
        expander = st.expander(label=f'Esempio #{i}')
        with expander:
            st.image(image, channels='BGR', caption= "Query image")
            result_original_dim = utils.merge_views(image, dst, H_file)
            st.image(result_original_dim, channels='BGR', caption= "Homography with original dimension")
            result = utils.merge_views(image, dst, fileNameHMatrix= H_file, dim = (current_src.shape[1], current_src.shape[0]))
            st.image(result, channels='BGR', caption= "Homography with same dimension of src")
        i = i+1


