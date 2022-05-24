import cv2
import numpy as np
from skimage.filters import sobel
from skimage.morphology import binary_closing, binary_erosion, binary_dilation
from .modules.identifier import get_objs_contours, crop_by_contour, find_max_contour
from .modules.placer import place


def check_image(img_path, scale_param):
    img = cv2.imread(img_path)[:, :, ::-1]
    sheet_with_poly, sheet_area, bottom_part = crop_image(img)
    poly_contour = get_poly_contour(sheet_with_poly)
    objects = find_objects(bottom_part)
    objs_contours = get_objs_contours(objects, sheet_area, scale_param)

    return place(sheet_with_poly.shape[:2], objs_contours, poly_contour)


def crop_image(image):
    b = image[..., 2]
    g = image[..., 1]
    mask = b > g

    binary_mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paper_ind = find_max_contour(contours)
    sheet_with_poly, bottomx, _, _, _ = crop_by_contour(image, contours, paper_ind, 40)

    bottom_part = image[bottomx + 10:, :]
    return sheet_with_poly, cv2.contourArea(contours[paper_ind]),   bottom_part


def get_poly_contour(sheet_with_poly):
    gray = cv2.cvtColor(sheet_with_poly, cv2.COLOR_RGB2GRAY)

    a = sobel(gray)
    mask = a > 0.01

    for i in range(7):
        mask = binary_erosion(mask)
        mask = binary_dilation(mask)
        mask = binary_closing(mask, selem=np.ones((6, 6)))

    binary_mask = mask.astype(np.uint8)

    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    er_kernel = np.ones((3, 3), np.uint8)

    binary_mask = cv2.erode(binary_mask, er_kernel)
    binary_mask = cv2.dilate(binary_mask, dil_kernel)
    binary_mask = cv2.dilate(binary_mask, dil_kernel)
    binary_mask = cv2.erode(binary_mask, er_kernel)
    binary_mask = cv2.dilate(binary_mask, dil_kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    poly_ind = find_max_contour(contours)

    cv2.drawContours(gray, contours, poly_ind, 255, -1)
    return contours[poly_ind]


def find_objects(img):
    r = img[..., 0]
    b = img[..., 2]
    g = img[..., 1]

    mask_y_1 = r > 125
    mask_y_2 = g < 140
    mask_y_3 = b < 45
    mask_y = mask_y_1 * mask_y_2 * mask_y_3

    mask_bl_1 = r < 90
    mask_bl_2 = b < 90
    mask_bl_3 = g < 90
    mask_bl = mask_bl_1 * mask_bl_2 * mask_bl_3

    mask1 = b > g
    mask2 = b > r

    mask = mask_bl + mask1 + mask2 + mask_y

    binary_mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_ = list(filter(lambda contour: cv2.contourArea(contour) > 1200, contours))

    objects = []

    for i in range(len(contours_)):
        found_obj, _, _, _, _ = crop_by_contour(img, contours_, i, 0)
        objects.append(found_obj)

    return objects
