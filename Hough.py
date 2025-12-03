################################################
#
# COMS30068 - filter2d.py
# TOPIC: create, save and display an image
#
# Getting-Started-File for OpenCV
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='Convert RGB to GRAY')
parser.add_argument('-name', '-n', type=str, default='coins2.png')
args = parser.parse_args()

# ==================================================
def sobelEdge(input):
    # intialise the output using the input
    edgeOutputX = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
    edgeOutputY = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
    # create the Gaussian kernel in 1D
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernelY = kernelX.T
    # we need to create a padded version of the input
    # or there will be border effects
    kernelRadiusX = round((kernelX.shape[0] - 1) / 2)
    kernelRadiusY = round((kernelX.shape[1] - 1) / 2)
    paddedInput = cv2.copyMakeBorder(input,
        kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
        cv2.BORDER_REPLICATE)
    # now we can do the convoltion
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            patch = paddedInput[i:i+kernelX.shape[0], j:j+kernelX.shape[1]]
            edgeOutputX[i, j] = (np.multiply(patch, kernelX)).sum()
            edgeOutputY[i, j] = (np.multiply(patch, kernelY)).sum()
    return edgeOutputX, edgeOutputY

# ==== MAIN ==============================================

imageName = args.name

# ignore if no such file is present.
if not os.path.isfile(imageName):
    print('No such file')
    sys.exit(1)

# Read image from file
image = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(image) is np.ndarray):
    print('Not image data')
    sys.exit(1)

# CONVERT COLOUR, BLUR AND SAVE
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = gray_image.astype(np.float32)

# apply sobel
edgemapX, edgemapY = sobelEdge(gray_image)
# magnitude
magnitude = np.sqrt(edgemapX**2 + edgemapY**2)
normmagnitude = (magnitude-magnitude.min())/(magnitude.max()-magnitude.min())
# orientation
anglemap = np.arctan2(edgemapY, edgemapX)
# edge
edgemap = normmagnitude > 0.2

rmin = 20
rmax = 100
hough3D = np.zeros([image.shape[0], image.shape[1], rmax-rmin+1], dtype=np.float32)
for i in range(0, image.shape[0]):  # go through all rows (or scanlines)
    for j in range(0, image.shape[1]):
        if edgemap[i, j] > 0:
            for r in range(rmin, rmax+1):
                x = (j + np.array([-1, 1])*r*np.cos(anglemap[i, j])).astype(int)
                y = (i + np.array([-1, 1])*r*np.sin(anglemap[i, j])).astype(int)
                for k in range(0, 2):
                    if (y[k] >= 0) and (y[k] < image.shape[0]) and (x[k] >= 0) and (x[k] < image.shape[1]):
                        hough3D[y[k], x[k], r-rmin] += 1

hough2D = np.sum(hough3D, axis=2)

threshold = 20  # Try to change the threshold
circle_parameters_ls = []
for i in range(0, hough3D.shape[0]):
    for j in range(0, hough3D.shape[1]):
        for k in range(0, hough3D.shape[2]):
            if hough3D[i, j, k] >= threshold:
                circle_parameters_ls.append([j, i, k + rmin])
# Plot the circles according to the parameters on the original image
imagewithcircle = image
for circle_parameters in circle_parameters_ls:
    imagewithcircle = cv2.circle(imagewithcircle,
                                 (circle_parameters[0], circle_parameters[1]),
                                 int(circle_parameters[2]),
                                 color=(0, 0, 255),
                                 thickness=2)
cv2.imwrite("imagewithcircle.jpg", imagewithcircle)

# save image
cv2.imwrite("edgemapX.jpg", (edgemapX-edgemapX.min())/(edgemapX.max()-edgemapX.min())*255)
cv2.imwrite("edgemapY.jpg", (edgemapY-edgemapY.min())/(edgemapY.max()-edgemapY.min())*255)
cv2.imwrite("hough2D.jpg", (hough2D-hough2D.min())/(hough2D.max()-hough2D.min())*255)

