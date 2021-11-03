import cv2
import numpy as np

image_path = "liz.jpg"
map_path = "liz-map.jpg"


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
    