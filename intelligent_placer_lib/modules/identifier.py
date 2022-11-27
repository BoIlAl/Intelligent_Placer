from sklearn.cluster import KMeans
from skimage.filters import gaussian, threshold_otsu
from skimage.color import rgb2gray
from collections import OrderedDict
import numpy as np
import cv2


def get_objs_contours(objects, sheet_area, scale_param):
    origin = list(OrderedDict.fromkeys(object_ident(objects)))
    origin_contours = []
    for item in origin:
        origin_contours.append(get_origin_contour(item, sheet_area, scale_param))
    return origin_contours


def object_ident(objects):
    origin = []
    for obj in objects:
        colors = primary_colors_count(obj, 3)
        if cat_check(colors):
            origin.append('bruh_cat')
        elif domino_check(colors):
            origin.append('domino')
        elif fish_check(colors):
            origin.append('fish')
        elif duck_check(colors):
            origin.append('duck')
        elif gamepad_check(colors):
            origin.append('gamepad')
        elif marker_check(colors):
            origin.append('marker')
        elif flash_check(colors):
            origin.append('flash_drive')
        elif pen_check(colors):
            origin.append('pen')
        elif chess_check(colors):
            origin.append('chess_figure')
        else:
            origin.append('earring')
    return origin


def primary_colors_count(obj, n_colors):
    img = obj.reshape((obj.shape[0] * obj.shape[1], 3))
    kmeans_clustering = KMeans(n_clusters=n_colors, n_init=5, max_iter=100)
    kmeans_clustering.fit(img)
    centers = kmeans_clustering.cluster_centers_.astype(np.uint8).tolist()
    centers = sorted(centers, key=lambda x: x[0])
    return centers


def cat_check(colors):
    if 105 < colors[0][0] < 130 and 55 < colors[0][1] < 80 and 10 < colors[0][2] < 35 and \
            125 < colors[1][0] < 155 and 70 < colors[1][1] < 105 and 25 < colors[1][2] < 45 and \
            140 < colors[2][0] < 170 and 90 < colors[2][1] < 130 and 40 < colors[2][2] < 75:
        return True
    return False


def fish_check(colors):
    if 55 < colors[0][0] < 80 and 45 < colors[0][1] < 70 and 30 < colors[0][2] < 50 and \
            80 < colors[1][0] < 105 and 70 < colors[1][1] < 90 and 50 < colors[1][2] < 70 and \
            125 < colors[2][0] < 150 and 100 < colors[2][1] < 125 and 55 < colors[2][2] < 75:
        return True
    return False


def domino_check(colors):
    if 40 < colors[0][0] < 75 and 35 < colors[0][1] < 70 and 30 < colors[0][2] < 65 and \
            80 < colors[1][0] < 115 and 65 < colors[1][1] < 110 and 45 < colors[1][2] < 95 and \
            135 < colors[2][0] < 170 and 130 < colors[2][1] < 165 and 125 < colors[2][2] < 160:
        return True
    return False


def duck_check(colors):
    if 100 < colors[0][0] < 135 and 75 < colors[0][1] < 105 and 15 < colors[0][2] < 45 and \
            125 < colors[1][0] < 160 and 100 < colors[1][1] < 135 and 25 < colors[1][2] < 85 and \
            135 < colors[2][0] < 170 and 115 < colors[2][1] < 150 and 20 < colors[2][2] < 50:
        return True
    return False


def gamepad_check(colors):
    if 30 < colors[0][0] < 50 and 25 < colors[0][1] < 45 and 25 < colors[0][2] < 45 and \
            55 < colors[1][0] < 90 and 45 < colors[1][1] < 75 and 45 < colors[1][2] < 70 and \
            120 < colors[2][0] < 150 and 95 < colors[2][1] < 125 and 50 < colors[2][2] < 90:
        return True
    return False


def chess_check(colors):
    if 75 < colors[0][0] < 95 and 50 < colors[0][1] < 70 and 20 < colors[0][2] < 45 and \
            95 < colors[1][0] < 115 and 65 < colors[1][1] < 85 and 30 < colors[1][2] < 55 and \
            110 < colors[2][0] < 130 and 85 < colors[2][1] < 105 and 45 < colors[2][2] < 65:
        return True
    return False


def marker_check(colors):
    if 30 < colors[0][0] < 60 and \
            35 < colors[1][0] < 70 and \
            90 < colors[2][0] < 135 and 80 < colors[2][1] < 125 and 60 < colors[2][2] < 100:
        return True
    return False


def pen_check(colors):
    if 45 < colors[0][0] < 95 and 95 < colors[0][1] < 135 and 65 < colors[0][2] < 110:
        return True
    return False


def flash_check(colors):
    if 35 < colors[0][0] < 70 and 35 < colors[0][1] < 75 and 55 < colors[0][2] < 105:
        return True
    return False


def get_origin_contour(name, sheet_area, scale_param):
    img = cv2.imread('./intelligent_placer_lib/modules/origin/' + name + '.jpg')[:, :, ::-1]
    b = img[..., 2]
    r = img[..., 0]
    if name == 'marker':
        mask1 = b > 150
    else:
        mask1 = b > r

    mask1_binary = mask1.astype(np.uint8)

    contours, _ = cv2.findContours(mask1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ind = find_max_contour(contours)

    cropped_img, _, _, _, _ = crop_by_contour(img, contours, ind, 40)

    a4_area = cv2.contourArea(contours[ind])
    scale = sheet_area / a4_area * scale_param

    img_blur = gaussian(cropped_img, sigma=1, multichannel=True)
    img_blur_gray = rgb2gray(img_blur)
    thresh_otsu = threshold_otsu(img_blur_gray)
    mask2 = img_blur_gray <= thresh_otsu

    mask2_binary = mask2.astype(np.uint8)

    contours, _ = cv2.findContours(mask2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ind = find_max_contour(contours)

    return scale_contour(contours[ind], scale)


def crop_by_contour(img, contours, ind, eps):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, ind, 255, -1)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_img = img[topx + eps: bottomx - eps, topy + eps: bottomy - eps]
    return cropped_img, bottomx, bottomy, topx, topy


def find_max_contour(contours):
    max_area = 0
    ind = 0
    for i in range(len(contours)):
        if (cv2.contourArea(contours[i])) > max_area:
            max_area = cv2.contourArea(contours[i])
            ind = i
    return ind


def scale_contour(contour, scale):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = contour - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled
