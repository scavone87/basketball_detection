import streamlit as st
import cv2
import numpy as np
import PIL
from constants import PATH_IMGS
from image_processing import draw_lines, draw_points, draw_grid, find_points, mean

quadrants_dict = {
    'Quadrante 1:1': [[0, 1/3, 0, 1/3],[0, 1/3, 1/7, 1/3]],
    'Quadrante 1:2': [[1/3, 2/3, 0, 1/3], [1/3, 2/3, 1/7, 1/3]],
    'Quadrante 1:3': [[2/3, 1, 0, 1/3], [2/3, 1, 1/7, 1/3]],
    'Quadrante 2:1': [[0, 1/3, 1/3, 2/3], [0, 1/3, 1/3, 0.55]],
    'Quadrante 2:2': [[1/3, 2/3, 1/3, 2/3], [1/3, 2/3, 1/3, 0.55]],
    'Quadrante 2:3': [[2/3, 1, 1/3, 2/3], [2/3, 1, 1/3, 0.55]],
    'Quadrante 3:1': [[0, 1/3, 2/3, 1], [0, 1/3, 0.55, 0.85]],
    'Quadrante 3:2': [[1/3, 2/3, 2/3, 1], [1/3, 2/3, 0.55, 0.85]],
    'Quadrante 3:3': [[2/3, 1, 2/3, 1], [2/3, 1, 0.55, 0.85]]
}


images_dict = {
  'Con giocatori': PATH_IMGS + "src_hough.jpg",
  'Con giocatori + mask': PATH_IMGS + "pills_src.png",
  'Senza giocatori': PATH_IMGS + "src1.jpg"
}



def check_points(points_img1, points_img2):
    if len(points_img1) == 0 or len(points_img2) == 0 or len(points_img1) < 4 or len(points_img2) < 4:
        message = "Non esistono abbastanza punti per l'omografia"
        st.error(message)
        return False
    return True

def execute_homography(points_img1, points_img2, src, dst, scelta):
    if scelta == 'Si':
        reprojThresh = st.slider("Ransac Reprojection threshold", min_value=1.0, max_value=10.0, step=0.5)
        (H, status) = cv2.findHomography(np.array(points_img2), np.array(points_img1), cv2.RANSAC, reprojThresh)
    else:
        (H, status) = cv2.findHomography(np.array(points_img2), np.array(points_img1))
    
    plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    # Crea una maschera booleana dove tutte le condizioni sono vere
    mask = np.all(plan_view == [0, 0, 0], axis=-1)
    # Utilizza la maschera per copiare i pixel da dst a plan_view
    plan_view[mask] = dst[mask]
    st.image(plan_view, channels="BGR")



st.set_page_config(page_title='Omografia con linee di Hough', page_icon=':basketball:', layout='centered', initial_sidebar_state='auto')   

st.title("Omografia con linee di Hough")

st.markdown('''<div style="text-align: justify"> In questo modulo è stata utilizzata una strategia che prevede l'utilizzo della trasformata di Hough.
E’ una tecnica che permette di riconoscere particolari configurazioni di punti presenti nell’immagine, come segmenti, curve o altre forme prefissate.
Il principio fondamentale è che la forma cercata può essere espressa tramite una funzione nota che fa uso di un insieme di parametri. </div>
<br> 
<div style="text-align: justify"> Nel caso del nostro progetto, l'obiettivo è quello di sfruttare la trasformata per ottenere delle linee le cui intersezioni restituiscano
dei punti candidati ad essere utilizzati per il calcolo dell'omografia. 
</div>
''', unsafe_allow_html=True)
st.markdown("---")
st.markdown("## <b>Applicazione</b>", unsafe_allow_html= True)
st.subheader("Linee di Hough su Immagine di Destination")
dst = cv2.imread(PATH_IMGS + 'dst.jpg', cv2.IMREAD_COLOR)
dst_copy = dst.copy() 

with st.form(key='hough_dst'):
    threshold= st.slider("Threshold", min_value=10, max_value=400, value=135, step=5)
    minLineLength=st.slider("Min Line Length", min_value=10, max_value=400, value=50, step=5)
    maxLineGap=st.slider("Max Line Gap", min_value=10, max_value=400, value=40, step=5)
    st.form_submit_button(label='Calcola')
    st.markdown("**Nota**: in base ai parametri selezionati, la libreria che consente di ricavare le intersezioni potrebbe non trovare punti per l'omografia. In tal caso modificare i valori degli sliders.")


points = draw_lines(dst_copy, 1, np.pi/180, threshold=threshold, min_line_length=minLineLength, max_line_gap=maxLineGap, mode='probabilistic')
draw_points(dst_copy, points)
st.image(dst_copy, channels="BGR")
draw_points(dst, points,mode='single')
cv2.imwrite(PATH_IMGS + 'final_dst.png', dst)
    


st.subheader("Linee di Hough su Immagine di Source")

with st.form(key='hough_src'):
    scelta = st.selectbox("Scegli un'immagine", images_dict.keys())
    threshold= st.slider("Threshold", min_value=10, max_value=400, value=135, step=5, key="t2")
    minLineLength=st.slider("Min Line Length", min_value=10, max_value=400, value=50, step=5, key="min2")
    maxLineGap=st.slider("Max Line Gap", min_value=10, max_value=400, value=40, step=5, key="max2")
    st.form_submit_button(label='Calcola')
    st.markdown("**Nota**: in base ai parametri selezionati, la libreria che consente di ricavare le intersezioni potrebbe non trovare punti per l'omografia. In tal caso modificare i valori degli sliders.")

src = cv2.imread(images_dict[scelta], cv2.IMREAD_COLOR)
src_copy = src.copy() 
src_points = draw_lines(src_copy, 1, np.pi/180, threshold=threshold, min_line_length=minLineLength, max_line_gap=maxLineGap, mode='probabilistic')
draw_points(src_copy, src_points)
st.image(src_copy, channels="BGR")
draw_points(src, src_points, mode='single')
cv2.imwrite(PATH_IMGS + 'final_src.png', src)


st.subheader("Divisione immagini in griglie")   
st.markdown('''
<div style="text-align: justify"> Come è possibile notare, è molto difficile riuscire ad ottenere gli stessi punti su entrambe le immagini.
Anche nel caso in cui si trovino punti in comune,  si avrebbe un numero di punti differenti tra le due immagini, rendendo impossibile il matching tra di essi. 
Quindi, la strategia adottata è quella di dividere le immagini in 9 settori mediante una griglia, analizzare un settore per volta e scegliere per ognuno di essi un punto trovato mediante euristica 
(media).
</div> <br>
''', unsafe_allow_html= True)

grid_src = draw_grid(src.copy())
grid_dst = cv2.imread(PATH_IMGS + "dst_grid.jpg")


image1 = PIL.Image.open(PATH_IMGS + "final_dst.png")
image2 = PIL.Image.open(PATH_IMGS + "final_src.png")

x = image1.size[0]
y = image1.size[1]

x1 = image2.size[0]
y1 = image2.size[1]

points_img1 = []
points_img2 = []

for list_value in quadrants_dict.values():
    par_img1 = list_value[0]
    par_img2 = list_value[1]  
    p_im1 = find_points(image1, int(x * par_img1[0]), int(x*par_img1[1]), int(y* par_img1[2]), int(y*par_img1[3]))
    p_im2 = find_points(image2, int(x1 * par_img2[0]), int(x1*par_img2[1]), int(y1* par_img2[2]), int(y1*par_img2[3]))
    if p_im1 != None and p_im2 != None:
        cv2.circle(grid_dst, (int(p_im1[0]), int(p_im1[1])), 1, (255,127,0), 12)
        cv2.circle(grid_src, (int(p_im2[0]), int(p_im2[1])), 1, (255,127,0), 12)
        points_img1.append(p_im1)
        points_img2.append(p_im2)
print(f'Points Image 1: {points_img1} \n Length: {len(points_img1)}')
print(f'Points Image 2: {points_img2} \n Length: {len(points_img2)}')

st.image(grid_src, channels = "BGR", caption="La griglia è stata costruita in modo da avere rettangoli bassi in alto e rettangoli alti in basso per via della prospettiva riproducendo quanto più fedelmente quella applicata all'immagine 2D.")
st.image(grid_dst, channels = "BGR")

st.subheader("Risultato")
scelta = st.radio("Utilizzo del RANSAC per gli outliers", ['Si', 'No'], index=1)

if check_points(points_img1, points_img2):
    execute_homography(points_img1, points_img2, src, dst, scelta)


