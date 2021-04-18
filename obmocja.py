import sys
import os
import numpy as np
import cv2 as cv

AREA_TRESHOLD = 70

def contour_area(cnt):
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    area = cv.contourArea(cnt)
    return area

# Retuns indices and areas
def filter_contours(contours):
    ret = []
    areas = []
    for i, cnt in enumerate(contours):
        area = contour_area(cnt)
        if (area > AREA_TRESHOLD):
            ret.append(i)
            areas.append(area)
    return ret, areas
    
# Pretvori .npy datoteko v sliko
# `np_img` je vsebina .npy datoteke
# .npy datoteko preberes: `np.load(file_path)`
def np_to_cv_img(np_img):
    return np.array(np_img.squeeze() * 255, dtype = np.uint8)
    
# @returns (list<contours>, list<indices>, list<areas>)
def najdi_poskodovana_obmocja(cv_gray_img):
    _, thresh = cv.threshold(cv_gray_img, 200, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    indices, areas = filter_contours(contours)
    return contours, indices, areas
    
if (__name__ == "__main__"):
    file_path = ""

    if ( len(sys.argv) > 1 ):
        file_path = sys.argv[1]
    else:
        raise NameError("Ni parametra")

    if (not file_path.endswith(".npy")):
        raise NameError("Ni .npy")
        
    float_img = np.load(file_path)
    
    imgray = np_to_cv_img(float_img)
    
    contours, indices, areas = najdi_poskodovana_obmocja(imgray)
    
    for i in indices:
        cv.drawContours(imgray, contours, i, (0,255,0), 3)

    cv.imshow(os.path.basename(file_path), imgray)

    cv.waitKey(0)
    cv.destroyAllWindows()