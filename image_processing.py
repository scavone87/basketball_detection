import cv2
import numpy as np
import poly_point_isect as bot

def detectAndDescribe(image, method=None):
    descriptor_dict = {
        'sift': cv2.SIFT_create,
        'brisk': cv2.BRISK_create,
        'orb': cv2.ORB_create
    }
    assert method in descriptor_dict, f"Invalid method. Possible values are: {', '.join(descriptor_dict.keys())}"
    descriptor = descriptor_dict[method]()
    kps, features = descriptor.detectAndCompute(image, None)
    return kps, features

def createMatcher(method, crossCheck):
    "Crea e restituisce un oggetto Matcher"

    if method in ['sift', 'surf']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method in ['orb', 'brisk']:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    else:
        raise ValueError("Metodo di feature matching non supportato.")

    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)

    # Esegue il matching dei descrittori
    matches = bf.match(featuresA, featuresB)

    # Ordina i match in base alla distanza
    matches = sorted(matches, key=lambda x: x.distance)

    print("Raw matches (Brute force):", len(matches))
    return matches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # Calcola i match grezzi e inizializza la lista dei match effettivi
    matches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(matches))
    good_matches = []

    for m, n in matches:
        # Verifica che la distanza sia entro il rapporto specificato
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # Calcola la matrice di omografia
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return matches, H, status
    else:
        return None


def draw_lines(img, rho, theta, threshold, min_line_length=None, max_line_gap=None, mode='standard'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    points = []

    if mode == 'standard':
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                x0 = cos_theta * rho
                y0 = sin_theta * rho
                line_length = 1000
                x1 = int(x0 + line_length * (-sin_theta))
                y1 = int(y0 + line_length * (cos_theta))
                x2 = int(x0 - line_length * (-sin_theta))
                y2 = int(y0 - line_length * (cos_theta))
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    elif mode == 'probabilistic':
        if min_line_length is None or max_line_gap is None:
            raise ValueError("Per la modalità 'probabilistic', devi fornire 'min_line_length' e 'max_line_gap'.")
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    else:
        raise ValueError("Il parametro 'mode' non è valido. Deve essere 'standard' o 'probabilistic'.")

    return points


def draw_points(img, points, mode='block'):
    try:
        intersections = bot.isect_segments(points)
    except AssertionError:
        print("Nessun punto trovato! Riprovare con altri valori!")
        return

    new_intersections = []
    for idx, inter in enumerate(intersections):
        a, b = inter
        match = False
        for other_inter in intersections[idx + 1:]:
            c, d = other_inter
            if abs(c - a) < 8 and abs(d - b) < 8:
                match = True
                break
        if not match:
            new_intersections.append(inter)

    if mode == 'block':
        for inter in new_intersections:
            a, b = inter
            img[int(b):int(b)+6, int(a):int(a)+6] = [0, 255, 0]
    elif mode == 'single':
        for inter in new_intersections:
            a, b = inter
            img[int(b), int(a)] = [0, 255, 0]
    else:
        raise ValueError("Il parametro 'mode' non è valido. Deve essere 'block' o 'single'.")


def mean(points):
    if len(points) == 0:
        return None
    x_sum = 0
    y_sum = 0
    for point in points:
        x_sum += point[0]
        y_sum += point[1]
    return [x_sum / len(points), y_sum / len(points)]

def find_points(img, x1, x2, y1, y2):
    points = []
    for y in range(y1, y2):
        for x in range(x1, x2):
            rgb_pixel_value = img.getpixel((x, y))
            if rgb_pixel_value == (0, 255, 0):
                points.append([x, y])
    if len(points) == 0:
        return None
    mean_points = mean(points)
    return mean_points


def draw_grid(img):
  width = img.shape[1]
  height = img.shape[0]
  for x in range(0, width, int(width/3)):
    cv2.line(img, (x, int(height/7)), (x, int(height*0.85)), (0, 255, 0), thickness=  3)
  cv2.line(img, (0,int(height/7)), (width, int(height/7)),(0, 255, 0), thickness = 3)
  #cv2.line(img, (0,height), (width, height),(0, 255, 0), thickness=  3)
  cv2.line(img, (0, int(height/3)), (width, int(height/3)),(0, 255, 0), thickness=  3)
  cv2.line(img, (0, int(height*0.55)), (width, int(height*0.55)),(0, 255, 0), thickness=  3)
  cv2.line(img, (0, int(height*0.85)), (width, int(height*0.85)),(0, 255, 0), thickness=  3)
  return img

