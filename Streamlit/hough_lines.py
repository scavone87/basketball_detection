import streamlit as st
import cv2
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL
from IPython.display import display
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from constants import PATH_IMGS
import poly_point_isect as bot

'''quadrants_dict = {
    'Quadrante 1:1': [[0, 1/3, 0, 1/3],[0, 1/3, 0, 1/6]],
    'Quadrante 1:2': [[1/3, 2/3, 0, 1/3], [1/3, 2/3, 0, 1/6]],
    'Quadrante 1:3': [[2/3, 1, 0, 1/3], [2/3, 1, 0, 1/6]],
    'Quadrante 2:1': [[0, 1/3, 1/3, 2/3], [0, 1/3, 1/6, 1/2]],
    'Quadrante 2:2': [[1/3, 2/3, 1/3, 2/3], [1/3, 2/3, 1/6, 1/2]],
    'Quadrante 2:3': [[2/3, 1, 1/3, 2/3], [2/3, 1, 1/6, 1/2]],
    'Quadrante 3:1': [[0, 1/3, 2/3, 1], [0, 1/3, 1/2, 1]],
    'Quadrante 3:2': [[1/3, 2/3, 2/3, 1], [1/3, 2/3, 1/2, 1]],
    'Quadrante 3:3': [[2/3, 1, 2/3, 1], [2/3, 1, 1/2, 1]]
}'''

quadrants_dict = {
    'Quadrante 1:1': [[0, 1/3, 0, 1/3],[0, 1/3, 0, 1/4]],
    'Quadrante 1:2': [[1/3, 2/3, 0, 1/3], [1/3, 2/3, 0, 1/4]],
    'Quadrante 1:3': [[2/3, 1, 0, 1/3], [2/3, 1, 0, 1/4]],
    'Quadrante 2:1': [[0, 1/3, 1/3, 2/3], [0, 1/3, 1/4, 3/5]],
    'Quadrante 2:2': [[1/3, 2/3, 1/3, 2/3], [1/3, 2/3, 1/4, 3/5]],
    'Quadrante 2:3': [[2/3, 1, 1/3, 2/3], [2/3, 1, 1/4, 3/5]],
    'Quadrante 3:1': [[0, 1/3, 2/3, 1], [0, 1/3, 3/5, 1]],
    'Quadrante 3:2': [[1/3, 2/3, 2/3, 1], [1/3, 2/3, 3/5, 1]],
    'Quadrante 3:3': [[2/3, 1, 2/3, 1], [2/3, 1, 3/5, 1]]
}

'''quadrants_dict = {
    'Quadrante 1:1': [[0, 1/3, 0, 1/3],[0, 1/3, 0, 1/3]],
    'Quadrante 1:2': [[1/3, 2/3, 0, 1/3], [1/3, 2/3, 0, 1/3]],
    'Quadrante 1:3': [[2/3, 1, 0, 1/3], [2/3, 1, 0, 1/3]],
    'Quadrante 2:1': [[0, 1/3, 1/3, 2/3], [0, 1/3, 1/3, 2/3]],
    'Quadrante 2:2': [[1/3, 2/3, 1/3, 2/3], [1/3, 2/3, 1/3, 2/3]],
    'Quadrante 2:3': [[2/3, 1, 1/3, 2/3], [2/3, 1, 1/3, 2/3]],
    'Quadrante 3:1': [[0, 1/3, 2/3, 1], [0, 1/3, 2/3, 1]],
    'Quadrante 3:2': [[1/3, 2/3, 2/3, 1], [1/3, 2/3, 1/2, 1]],
    'Quadrante 3:3': [[2/3, 1, 2/3, 1], [2/3, 1, 2/3, 1]]
}'''


images_dict = {
  'Con giocatori': PATH_IMGS + "src_hough.jpg",
  'Con giocatori + mask': PATH_IMGS + "pills_src.png",
  'Senza giocatori': PATH_IMGS + "src1.jpg"
}

def app():
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
        points = draw_lines_p(dst_copy, 1, np.pi/180, threshold=threshold, min_line_length=minLineLength, max_line_gap=maxLineGap)
        draw_points(dst_copy, points)
        st.image(dst_copy, channels="BGR")
        color_single_pixel(dst, points)
        cv2.imwrite(PATH_IMGS + 'final_dst.png', dst)
        st.markdown("**Nota**: in base ai parametri selezionati, la libreria che consente di ricavare le intersezioni potrebbe rilanciare un'eccezione se queste non vengono trovate")
        


    st.subheader("Linee di Hough su Immagine di Source")
    
    with st.form(key='hough_src'):
        scelta = st.selectbox("Scegli un'immagine", images_dict.keys())
        src = cv2.imread(images_dict[scelta], cv2.IMREAD_COLOR)
        src_copy = src.copy() 
        threshold= st.slider("Threshold", min_value=10, max_value=400, value=150, step=5, key="t2")
        minLineLength=st.slider("Min Line Length", min_value=10, max_value=400, value=50, step=5, key="min2")
        maxLineGap=st.slider("Max Line Gap", min_value=10, max_value=400, value=40, step=5, key="max2")
        st.form_submit_button(label='Calcola')
        src_points = draw_lines_p(src_copy, 1, np.pi/180, threshold=threshold, min_line_length=minLineLength, max_line_gap=maxLineGap)
        draw_points(src_copy, src_points)
        st.image(src_copy, channels="BGR")
        color_single_pixel(src, src_points)
        cv2.imwrite(PATH_IMGS + 'final_src.png', src)
        st.markdown("**Nota**: in base ai parametri selezionati, la libreria che consente di ricavare le intersezioni potrebbe rilanciare un'eccezione se queste non vengono trovate")



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
            print(int(p_im1[0]))
            cv2.circle(grid_dst, (int(p_im1[0]), int(p_im1[1])), 1, (255,127,0), 12)
            cv2.circle(grid_src, (int(p_im2[0]), int(p_im2[1])), 1, (255,127,0), 12)
            points_img1.append(p_im1)
            points_img2.append(p_im2)
    print(f'Points Image 1: {points_img1} \n Length: {len(points_img1)}')
    print(f'Points Image 2: {points_img2} \n Length: {len(points_img2)}')
    
    st.image(grid_src, channels = "BGR", caption="La griglia è stata costruita in modo da avere rettangoli bassi in alto e rettangoli alti in basso per via della prospettiva")
    st.image(grid_dst, channels = "BGR")

    st.subheader("Risultato")
    
    scelta = st.radio("Utilizzo del RANSAC per gli outliers", ['Si', 'No'], index=1)

    if scelta == 'Si':
        reprojThresh = st.slider("Ransac Reprojection threshold", min_value=1.0, max_value=10.0, step=0.5)
        (H, status) = cv2.findHomography(np.array(points_img2), np.array(points_img1), cv2.RANSAC,reprojThresh)
    else:
        (H, status) = cv2.findHomography(np.array(points_img2), np.array(points_img1))
    plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    for i in range(0,dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if plan_view.item(i,j,0) == 0 and plan_view.item(i,j,1) == 0 and plan_view.item(i,j,2) == 0:
                plan_view.itemset((i,j,0),dst.item(i,j,0))
                plan_view.itemset((i,j,1),dst.item(i,j,1))
                plan_view.itemset((i,j,2),dst.item(i,j,2))

    st.image(plan_view, channels="BGR")




def draw_lines_p(img, rho, theta, threshold, min_line_length, max_line_gap):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 200)
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
  # Disegna le linee sull'immagine
  points = []
  for line in lines:
    for x1, y1, x2, y2 in line:
        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2) 
  return points

def draw_lines(img, rho, theta, threshold):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 200)
  # Disegna le linee sull'immagine
  lines = cv2.HoughLines(edges,rho,theta,threshold)
  points = []
  for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
  return points

def draw_points(img, points):
  intersections = bot.isect_segments(points)
  for idx, inter in enumerate(intersections):
    a, b = inter
    match = 0
    for other_inter in intersections[idx:]:
        c, d = other_inter
        if abs(c-a) < 8 and abs(d-b) < 8:
            match = 1
            if other_inter in intersections:
                intersections.remove(other_inter)
                intersections[idx] = ((c+a)/2, (d+b)/2)
    if match == 0:
        intersections.remove(inter)
  for inter in intersections:
    a, b = inter
    for i in range(6):
        for j in range(6):
            img[int(b) + i, int(a) + j] = [0, 255, 0]

def color_single_pixel(img, points):
  intersections = bot.isect_segments(points)
  for idx, inter in enumerate(intersections):
    a, b = inter
    match = 0
    for other_inter in intersections[idx:]:
        c, d = other_inter
        if abs(c-a) < 8 and abs(d-b) < 8:
            match = 1
            if other_inter in intersections:
                intersections.remove(other_inter)
                intersections[idx] = ((c+a)/2, (d+b)/2)
    if match == 0:
        intersections.remove(inter)
  for inter in intersections:
    a, b = inter
    img[int(b), int(a)] = [0, 255, 0]

def mean(points):
  x = points[:, 0].mean()
  y = points[:, 1].mean()
  return [x, y]

def find_points(img, x1, x2, y1, y2):
  #print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
  #st.image(img.crop((x1,y1,x2,y2)), channels="BGR")
  points = []
  for y in range(y1, y2):   
    for x in range(x1, x2):
      rgb_pixel_value = img.getpixel((x, y))
      if rgb_pixel_value == (0, 255, 0):
        #print(str(x) + "," + str(y))
        points.append([x, y])
  print(points)
  if len(points) == 0:
    return
  list_points = np.array(points)
  mean_points = mean(list_points)
  return mean_points

def draw_grid(img):
  width = img.shape[1]
  height = img.shape[0]
  for x in range(0, width, int(width/3)):
    cv2.line(img, (x, 0), (x, height), (0, 255, 0), thickness=  3)
  cv2.line(img, (0,0), (width, 0),(0, 255, 0), thickness = 3)
  cv2.line(img, (0,height), (width, height),(0, 255, 0), thickness=  3)
  cv2.line(img, (0, int(height/4)), (width, int(height/4)),(0, 255, 0), thickness=  3)
  cv2.line(img, (0, int(height*3/5)), (width, int(height*3/5)),(0, 255, 0), thickness=  3)
  cv2.line(img, (0, height), (width, height),(0, 255, 0), thickness=  3)
  return img
