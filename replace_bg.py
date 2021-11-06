import cv2
import numpy as np


def bg_size_match(dim, bg_img):
    bg_dim = bg_img.shape

    if bg_dim[0] == dim[0] and bg_dim[1] == dim[1]:
        return bg_img
    elif bg_dim[0] > dim[0] and bg_dim[1] > dim[1]:
        offset_x = int((bg_dim[1] - dim[1]) / 2)
        offset_y = int((bg_dim[0] - dim[0]) / 2)
        return bg_img[offset_y:offset_y+dim[0], offset_x:offset_x+dim[1]]
    elif bg_dim[0] < dim[0] and (bg_dim[1] > dim[1] or bg_dim[1] == dim[1]):
        bg_img = cv2.resize(bg_img, (bg_dim[1], dim[0]))
        offset_x = int((bg_dim[1] - dim[1]) / 2)
        return bg_img[0:dim[0], offset_x:offset_x+dim[1]]
    elif (bg_dim[0] > dim[0] or bg_dim[0] == dim[0]) and bg_dim[1] < dim[1]:
        bg_img = cv2.resize(bg_img, (dim[1], bg_dim[0]))
        offset_y = int((bg_dim[0] - dim[0]) / 2)
        return bg_img[offset_y:offset_y+dim[0], 0:dim[1]]
    else:
        return cv2.resize(bg_img, (dim[1], dim[0]))


def normalize_map(map):
    n_map = np.zeros((map.shape[0], map.shape[1]))

    for row in range(len(map)):
        for col in range(len(map[0])):
            n_map[row, col] = (map[row, col] / 255)

    return n_map


def combine(img, map, bg):
    # map = normalize_map(map) # Map is already normalized as neural net output
    bg = bg_size_match(img.shape, bg)

    for row in range(len(img)):
        for col in range(len(img[row])):
            fg_pix = img[row, col]
            bg_pix = bg[row, col]
            weight = map[row, col]
            i_weight = 1-weight

            img[row, col] = (fg_pix * weight) + (bg_pix * i_weight)

    return img
