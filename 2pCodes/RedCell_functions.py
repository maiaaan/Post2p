import matplotlib.pyplot as plt
import os.path
import cv2 as cv
import numpy as np
import os
from PIL import Image as im
from scipy.ndimage import shift
import easygui
import functions

def DetectRedCellMask(bath_path,save_red_results,  min_area = 35 , max_area= 150):
    image =  os.path.join(bath_path, "red.tif")
    detected_mask_path = os.path.join(save_red_results, "red_masked.jpg")
    gray_image_path = os.path.join(save_red_results, "grey_image.jpg")
    # Load image
    img = cv.imread(image, cv.IMREAD_COLOR)
    output = img.copy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray)

    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)
    kernel = np.ones((4,4), np.uint8)

    dilation = cv.dilate(thresh, kernel, iterations=1)
    contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            cv.drawContours(output, [contour], -1, (0, 255, 0), 1)

    blank3 = nzero_array = np.zeros((512, 512),  dtype='uint8')
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            cv.drawContours(blank3, [contour], -1, 2, -1)

    cv.imwrite(detected_mask_path, output)
    cv.imwrite(gray_image_path, clahe_output)
    return blank3


def loadred(Base_path):
    suite2p_path = os.path.join(Base_path, "suite2p", "plane0")
    ops = np.load((os.path.join(suite2p_path, "ops.npy")), allow_pickle=True).item()
    Mean_image = ((ops['meanImg']))
    cell = np.load((os.path.join(suite2p_path, "iscell.npy")), allow_pickle=True)
    stat = np.load((os.path.join(suite2p_path, "stat.npy")), allow_pickle=True)
    single_red = cv.imread((os.path.join(Base_path, 'red.tif')))
    return suite2p_path, ops, Mean_image, cell, stat, single_red

def totalMask(cells, stat, ops):
    cell_info,_ = functions.detect_cell(cells, stat)
    neumask = np.zeros((ops['Ly'], ops['Lx']))
    for i in range(len(cell_info)):
        neumask[cell_info[i]['ypix'], cell_info[i]['xpix']] = 2
    return neumask, cell_info

def single_mask(ops, cell_info):
    separete_masks = []
    for i in range(0, len(cell_info)):
        neumask1 = np.zeros((ops['Ly'], ops['Lx']))
        neumask1[cell_info[i]['ypix'], cell_info[i]['xpix']] = 2
        separete_masks.append(neumask1)
    return separete_masks


def select_mask(save_red_results, thresh2, separete_masks, cell_true = 2):
    KeepMask = []
    comen_cell = []
    only_green_mask = []
    only_green_cell = []

    for i in range(len(separete_masks)):
        blank2 = np.zeros(thresh2.shape, dtype='uint8')
        blank2 = thresh2 + separete_masks[i]
        if (cell_true + 2) in blank2:
            comen_cell.append(i)
            KeepMask.append(separete_masks[i])
        else:
            only_green_cell.append(i)
            only_green_mask.append(separete_masks[i])
    print(len(KeepMask), 'common cell detected')
    save_direction1 = os.path.join(save_red_results, 'red_green_cells.npy')
    save_direction2 = os.path.join(save_red_results, 'only_green.npy')
    np.save(save_direction1, comen_cell, allow_pickle=True)
    np.save(save_direction2, only_green_cell, allow_pickle=True)
    return only_green_mask, only_green_cell, comen_cell, KeepMask, blank2



def total_detected_mask(KeepMask):
    blank3 = np.zeros(KeepMask[0].shape, dtype='uint8')
    for i in range(len(KeepMask)):
        blank3 = blank3 + KeepMask[i]
    return blank3

#RUN
# Base_path, ops, Mean_image, cell, stat, single_red = loadred()
#image_path =  os.path.join(Base_path, "grey_image.jpg")
# cell_info,_ = functions.detect_cell(cell, stat)
# separete_masks = single_mask(ops, cell_info)
# blank_image =  DetectRedCellMask(image_path)
# only_green_mask, only_green_cell, comen_cell, KeepMask, blank2 = select_mask(Base_path, blank_image, separete_masks, cell_true=2)