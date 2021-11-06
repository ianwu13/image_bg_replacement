import cv2
from os.path import exists

from bg_map import *
from replace_bg import *


def get_images():
    cov_img_name = input("Enter foreground image path (enter nothing to use default)\n")
    if (cov_img_name == ""):
        cov_img_name = "./images/sample_fg.jpg"
    if (not exists(cov_img_name)):
        print("Foreground image could not be found")
        quit()
    fg_img = cv2.imread(cov_img_name)
    
    hid_img_name = input("Enter background image path (enter nothing to use default)\n")
    if (hid_img_name == ""):
        hid_img_name = "./images/sample_bg.jpg"
    if (not exists(hid_img_name)):
        print("Background image could not be found")
        quit()
    bg_img = cv2.imread(hid_img_name)
    return fg_img, bg_img


def main():
    fg_img, bg_img = get_images()
    map = get_bg_map(fg_img)

    cv2.imshow("Foreground Image", fg_img)
    cv2.imshow("Saliency Map", map)
    cv2.imshow("Background Image", bg_img)

    out_img = combine(fg_img, map, bg_img)

    cv2.imshow("Combined Images", out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()