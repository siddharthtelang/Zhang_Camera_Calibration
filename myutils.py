import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math
import os
import scipy.optimize
square_side = 12.5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def getImagesPoints(imgs, h, w):
    images = imgs.copy()
    all_corners = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            corners2 = corners2.reshape(-1,2)
            # corners2 = np.hstack((corners2.reshape(-1,2), np.ones((corners2.shape[0], 1))))
            all_corners.append(corners2)
    return all_corners

def displayCorners(images, all_corners, h, w, save_folder):
    for i, image in enumerate(images):
        corners = all_corners[i]
        corners = np.float32(corners.reshape(-1, 1, 2))
        cv2.drawChessboardCorners(image, (w, h), corners, True)
        img = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
        cv2.imshow('frame', img)
        filename = os.path.join(save_folder, str(i)+ "draw.png")
        save_folder + str(i) + "draw.png"
        cv2.imwrite(filename, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def getWorldPoints(square_side, h, w):
    # h, w = [6, 9]
    Yi, Xi = np.indices((h, w)) 
    offset = 0
    lin_homg_pts = np.stack(((Xi.ravel() + offset) * square_side, (Yi.ravel() + offset) * square_side)).T
    return lin_homg_pts

def getH(src, dst):
    nrows = src.shape[0]
    if (nrows < 4):
        print("Need atleast four points to compute SVD.")
        return 0

    x = src[:, 0]
    y = src[:, 1]
    xp = dst[:, 0]
    yp = dst[:,1]
    A = []
    for i in range(nrows):
        row1 = np.array([x[i], y[i], 1, 0, 0, 0, -x[i]*xp[i], -y[i]*xp[i], -xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, x[i], y[i], 1, -x[i]*yp[i], -y[i]*yp[i], -yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, V = np.linalg.svd(A, full_matrices=True)
    H = V[-1, :].reshape((3, 3))
    H = H / H[2,2]
    return H

def getAllH(pixel_corners, world_corners):
    all_H = []
    for pixel_corner in pixel_corners:
        H = getH(world_corners, pixel_corner)
        # H, _ = cv2.findHomography(world_corners, pixel_corner, cv2.RANSAC, 5.0)
        all_H.append(H)
    return all_H

def getVij(hi, hj):
    Vij = np.array([ hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2] ])
    return Vij.T

def getV(all_H):
    v = []
    for H in all_H:
        h1 = H[:,0]
        h2 = H[:,1]

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11 - v22).T)
    return np.array(v)

def arrangeB(b):
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]
    return B

def getB(all_H):
    v = getV(all_H)
    # vb = 0
    U, sigma, V = np.linalg.svd(v)
    b = V[-1, :]
    print("B matrix is ", b)
    B = arrangeB(b)  
    return B

def getA(B):
    v0 = (B[0,1] * B[0,2] - B[0,0] * B[1,2])/(B[0,0] * B[1,1] - B[0,1]**2)
    lamb = B[2,2] - (B[0,2]**2 + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2]))/B[0,0]
    alpha = np.sqrt(lamb/B[0,0])
    beta = np.sqrt(lamb * (B[0,0]/(B[0,0] * B[1,1] - B[0,1]**2)))
    gamma = -(B[0,1] * alpha**2 * beta) / lamb 
    u0 = (gamma * v0 / beta) - (B[0,2] * alpha**2 / lamb)

    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return A

def getRotationAndTranslation(A, all_H):
    all_RT = []
    for H in all_H:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lambDa = np.linalg.norm(np.dot(np.linalg.inv(A), h1), 2)

        r1 = np.dot(np.linalg.inv(A), h1) / lambDa
        r2 = np.dot(np.linalg.inv(A), h2) / lambDa
        r3 = np.cross(r1, r2)
        t = np.dot(np.linalg.inv(A), h3) / lambDa
        
        RT = np.vstack((r1, r2, r3, t)).T
        all_RT.append(RT)

    return all_RT   

def loadImages(folder_name):
    files = os.listdir(folder_name)
    print("Loading images from ", folder_name)
    images = []
    for f in files:
        image_path = os.path.join(folder_name, f)
        # print(image_path)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def extractParamFromA(init_A, init_kc):
    alpha = init_A[0,0]
    gamma = init_A[0,1]
    beta = init_A[1,1]
    u0 = init_A[0,2]
    v0 = init_A[1,2]
    k1 = init_kc[0]
    k2 = init_kc[1]

    x0 = np.array([alpha, gamma, beta, u0, v0, k1, k2])
    return x0

def retrieveA(x0):
    alpha, gamma, beta, u0, v0, k1, k2 = x0
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]).reshape(3,3)
    kc = np.array([k1, k2])
    return A, kc

def lossFunc(params, init_all_RT, all_image_corners, world_corners):

    A, kc = retrieveA(params)
    alpha, gamma, beta, u0, v0, k1, k2 = params

    error_mat = []

    for i, image_corners in enumerate(all_image_corners): # for all images

        #RT for 3d world points
        RT = init_all_RT[i]
        #get ART for 2d world points - remove Z - set 0 and form the matrix again
        RT3 = np.array([RT[:,0], RT[:,1], RT[:,3]]).reshape(3,3)
        RT3 = RT3.T
        ART3 = np.dot(A, RT3)

        image_total_error = 0

        for j in range(world_corners.shape[0]):

            world_point_2d = world_corners[j]
            world_point_2d_homo = np.array([world_point_2d[0], world_point_2d[1], 1]).reshape(3,1)
            world_point_3d_homo = np.array([world_point_2d[0], world_point_2d[1], 0, 1]).reshape(4,1)

            #get radius of distortion
            XYZ = np.dot(RT, world_point_3d_homo)
            x =  XYZ[0] / XYZ[2]
            y = XYZ[1] / XYZ[2]
            # x = alpha * XYZ[0] / XYZ[2] #assume gamma as 0 
            # y = beta * XYZ[1] / XYZ[2] #assume gamma as 0
            r = np.sqrt(x**2 + y**2) #radius of distortion

            #observed image co-ordinates
            mij = image_corners[j]
            mij = np.array([mij[0], mij[1], 1], dtype = 'float').reshape(3,1)

            #projected image co-ordinates
            uvw = np.dot(ART3, world_point_2d_homo)
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]

            u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
            v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)

            mij_dash = np.array([u_dash, v_dash, 1], dtype = 'float').reshape(3,1)

            #get error
            e = np.linalg.norm((mij - mij_dash), ord=2)
            image_total_error = image_total_error + e

        error_mat.append(image_total_error / world_corners.shape[0])
    
    return np.array(error_mat)

