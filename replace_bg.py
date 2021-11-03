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


def map_normalization(map):

    print()


def combine():
    img = cv2.imread(image_path)

    map = cv2.imread(map_path)
    norm_image = cv2.normalize(map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    for row in range(len(img)):
        for col in range(len(img[row])):
            float_val = img[row, col]
            img[row, col] = (float_val * norm_image[row, col])
    
    cv2.imshow("removed-bg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
