import cv2
# from bg_map import *
from replace_bg import *


def main():
    img = cv2.imread("./images/dog.jpg")
    map = cv2.imread("./images/dog-map.jpg")
    bg = cv2.imread("./images/sample_bg.jpg")
    cv2.imshow("ORIGINAL", img)
    cv2.imshow("MAP", map)
    cv2.imshow("NEW BG", bg)
    cv2.imshow("OUTPUT", combine(img, map, bg))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()