import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math
import os
import argparse
import scipy.optimize
from myutils import *

def main(args):
    folder_name = args['folder_name']
    save_folder = args['save_folder']
    images = loadImages(folder_name)
    h, w = args["height"], args["width"] #squares

    all_image_corners = getImagesPoints(images, h, w)

    world_corners = getWorldPoints(square_side, h, w)
    displayCorners(images, all_image_corners, h, w, save_folder)

    print('Finding Homography from world coordinates to pixel coordinates')
    all_H_init = getAllH(all_image_corners, world_corners)

    # calculate B as per paper
    print("Calculating B matrix as per paper")
    B_init = getB(all_H_init)
    print("Estimated B = \n", B_init)

    print("Calculating A from B")
    A_init = getA(B_init)
    print("Initialized A = \n",A_init)

    print("Calculating rotation and translation from A")
    all_RT_init = getRotationAndTranslation(A_init, all_H_init)

    print("Init kc (distortion coefficient)")
    kc_init = np.array([0,0])
    print("Initialized kc = \n", kc_init)

    print('Start optimization')
    params = extractParamFromA(A_init, kc_init)
    res = scipy.optimize.least_squares(fun=lossFunc, x0=params, method="lm", args=[all_RT_init, all_image_corners, world_corners])

    params_optimized = res.x
    A_K = retrieveA(params_optimized)
    A_final = A_K[0]
    kc_final = A_K[1]

    K = np.array(A_final, np.float32).reshape(3,3)
    D = np.array([kc_final[0], kc_final[1], 0, 0] , np.float32)
    print('Camera Intrinsic Matrix K:\n', K)
    print('\nCamera Distortion Matrix D:\n', D)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder_name", "--folder_name", required=False,
            type=str,
            default=r"C:\Users\siddh\Documents\733\Zhang_Camera_Calibration\Data\Calibration_Imgs")
    parser.add_argument("-save_folder", "--save_folder", required=False,
            type=str,
            default=r"C:\Users\siddh\Documents\733\Zhang_Camera_Calibration\Results")
    parser.add_argument("-height", "--height", required=False, type=int, default=6)
    parser.add_argument("-width", "--width", required=False, type=int, default=9)
    args = vars(parser.parse_args())
    main(args)
