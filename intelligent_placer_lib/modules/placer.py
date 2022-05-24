import cv2
import numpy as np


def place(size, contours, polygon_contour):
    contours = sorted(contours, key=lambda x: cv2.minEnclosingCircle(x)[1], reverse=True)
    polygon_radius = cv2.minEnclosingCircle(polygon_contour)[1]
    polygon_area = cv2.contourArea(polygon_contour)
    area_all_items = 0
    for contour in contours:
        item_area = cv2.contourArea(contour)
        area_all_items += item_area
        if cv2.minEnclosingCircle(contour)[1] > polygon_radius or item_area > polygon_area:
            return False, []
    if area_all_items > polygon_area:
        return False, []

    x, y, w, h = cv2.boundingRect(polygon_contour)
    img_placement = np.zeros((size[0], size[1], 3), np.uint8)
    cv2.fillPoly(img_placement, [polygon_contour], (255, 255, 255))
    step = int(min(w, h) / 10)
    step_angle = np.pi / 6

    return try_place(x, y, w, h, img_placement, contours, step, step_angle, 0)


def try_place(x, y, w, h, img_placement, contours, step, step_angle, ind):
    if ind == len(contours):
        return True, img_placement

    xc, yc, wc, hc = cv2.boundingRect(contours[ind])
    contour = move_contour(contours[ind], -xc + x, -yc + y)
    eps = (w - 2 * int(wc / 2)) % step
    for y_place in range(y + int(hc / 2), y + h - int(hc / 2), step):
        for x_place in range(x + int(wc / 2), x + w - int(wc / 2), step):
            angle = 0
            while angle < 2 * np.pi:
                rot_contour = rotate_contour(contour, angle)
                if is_place(rot_contour, img_placement):
                    img_placement_copy = img_placement.copy()
                    cv2.fillPoly(img_placement_copy, [rot_contour], (0, 0, 0))
                    ans, res = try_place(x, y, w, h, img_placement_copy, contours, step, step_angle, ind + 1)
                    if ans:
                        return True, res
                angle += step_angle
            contour = move_contour(contour, step, 0)
        contour = move_contour(contour, 2 * int(wc / 2) - w + eps - step, step)
    return False, []


def move_contour(contour, step_x, step_y):
    return contour + [int(step_x), int(step_y)]


def rotate_contour(contour, angle):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    new_contour = move_contour(contour, -cx, -cy)
    ox, oy = new_contour[:, 0, 0], new_contour[:, 0, 1]
    new_x = np.hypot(ox, oy) * np.cos(np.arctan2(oy, ox) + angle)
    new_y = np.hypot(ox, oy) * np.sin(np.arctan2(oy, ox) + angle)
    new_contour[:, 0, 0] = new_x
    new_contour[:, 0, 1] = new_y
    new_contour = move_contour(new_contour, cx, cy)
    return new_contour


def is_place(contour, img_placement):
    img_contour = img_placement.copy()
    cv2.fillPoly(img_contour, [contour], (255, 255, 255))
    return np.all(np.logical_xor(np.logical_not(img_contour), img_placement))
